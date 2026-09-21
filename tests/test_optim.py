"""The PyTorch front end: Algorithm 1 end to end on small models."""
import pytest
import torch

from pcd import PCD

torch.set_default_dtype(torch.float64)


def make_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 3))


def losses_for(model, data_seed=0, scale2=1.0):
    gen = torch.Generator().manual_seed(data_seed)
    x, y = torch.randn(64, 8, generator=gen), torch.randint(0, 3, (64,), generator=gen)
    primary = torch.nn.functional.cross_entropy(model(x), y)
    secondary = scale2 * sum(p.abs().sum() for n, p in model.named_parameters() if n.endswith("weight"))
    return [primary, secondary]


def flat_grad(model, loss):
    gs = torch.autograd.grad(loss, list(model.parameters()), retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(model.parameters(), gs)])


def written(model):
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()])


@pytest.mark.parametrize("tau", [0.0, 0.05, 0.3, 1.0])
def test_first_step_matches_algorithm_1(tau):
    model = make_model()
    losses = losses_for(model)
    g1, g2 = flat_grad(model, losses[0]), flat_grad(model, losses[1])
    PCD(model.parameters(), tau=tau).backward(losses)
    # At t = 1 the bias-corrected EMA is exactly ||g||^2, so s_i = 1 / sqrt(||g_i||^2 + eps).
    t1, t2 = (g / torch.sqrt(g @ g + 1e-8) for g in (g1, g2))
    d = t1 if t2 @ t1 >= tau * t2 @ t2 else t1 + (tau - t2 @ t1 / (t2 @ t2)) * t2   # Cor. 4.6
    assert torch.allclose(written(model), d * g1.norm() / d.norm(), atol=1e-12)


def test_rescale_keeps_direction_and_primary_norm():
    model = make_model()
    losses = losses_for(model, data_seed=3)
    g1, g2 = flat_grad(model, losses[0]), flat_grad(model, losses[1])
    tau = 0.4
    info = PCD(model.parameters(), tau=tau).backward(losses)
    d = written(model)
    t1, t2 = (g / torch.sqrt(g @ g + 1e-8) for g in (g1, g2))
    d_tilde = t1 + info.mu[0] * t2                        # the analyzed direction (Thm. 4.2)
    assert torch.isclose(d.norm(), g1.norm())
    assert torch.allclose(d / d.norm(), d_tilde / d_tilde.norm(), atol=1e-12)
    assert t2 @ d_tilde >= tau * t2 @ t2 - 1e-12          # per-step secondary progress (Prop. 4.4)


def test_scale_invariance_of_the_update():
    """Multiplying a secondary loss by a constant leaves the update unchanged (Thm. B.9, up to eps)."""
    model_a, model_b = make_model(), make_model()
    PCD(model_a.parameters(), tau=0.2).backward(losses_for(model_a, scale2=1.0))
    PCD(model_b.parameters(), tau=0.2).backward(losses_for(model_b, scale2=1e4))
    assert torch.allclose(written(model_a), written(model_b), rtol=1e-6, atol=1e-10)


def test_primary_stationary_halts():
    w = torch.nn.Parameter(torch.randn(5))
    info = PCD([w], tau=0.5).backward([(w * 0).sum(), w.abs().sum()])   # g_1 = 0
    assert torch.count_nonzero(w.grad) == 0 and info.mu == [0.0]


def test_parameters_no_loss_depends_on_are_skipped():
    a, b, unused = (torch.nn.Parameter(torch.randn(3)) for _ in range(3))
    unused.grad = torch.ones(3)                           # stale gradient from elsewhere
    PCD([a, b, unused], tau=0.1).backward([(a ** 2).sum(), b.abs().sum()])
    assert unused.grad is None and a.grad is not None and b.grad is not None


def test_three_objectives_and_diagnostics():
    torch.manual_seed(1)
    model = torch.nn.Linear(10, 4)
    x, y = torch.randn(32, 10), torch.randint(0, 4, (32,))
    losses = [torch.nn.functional.cross_entropy(model(x), y),
              model.weight.abs().sum(),
              torch.linalg.svdvals(model.weight).sum()]
    info = PCD(model.parameters(), tau=0.3).backward(losses)
    assert len(info.mu) == 2 and len(info.cosine) == 2 and len(info.scale) == 3
    assert info.feasible and all(m >= 0 for m in info.mu) and set(info.active) <= {1, 2}


def test_state_dict_round_trip():
    model, clone = make_model(), make_model()             # identical weights
    pcd = PCD(model.parameters(), tau=0.1)
    for seed in range(3):                                 # build up some EMA state
        pcd.backward(losses_for(model, data_seed=seed))
    fresh = PCD(clone.parameters(), tau=0.9)
    fresh.load_state_dict(pcd.state_dict())
    pcd.backward(losses_for(model, data_seed=11))
    fresh.backward(losses_for(clone, data_seed=11))
    assert fresh.tau == 0.1 and torch.equal(written(model), written(clone))


def test_training_loop_with_an_optimizer():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    pcd = PCD(model.parameters(), tau=0.1)
    x = torch.randn(16, 4)
    for _ in range(3):
        opt.zero_grad()
        pcd.backward([model(x).pow(2).mean(), model.weight.abs().sum()])
        opt.step()
    assert torch.isfinite(model.weight).all()
