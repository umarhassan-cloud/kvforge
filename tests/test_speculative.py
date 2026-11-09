import pytest
import torch

from kvforge import InferenceEngine, SamplingParams, SpeculativeConfig, build_model
from kvforge.model_loader import PRESETS
from kvforge.speculative import make_layer_skip_draft, rejection_sample

from .conftest import dense_greedy, make_config


def test_rejection_sampler_preserves_target_distribution():
    """Leviathan et al. Thm 1: the first emitted token is distributed exactly as p,
    regardless of the draft distribution q."""
    torch.manual_seed(0)
    vocab, k, trials = 6, 3, 200_000
    p = torch.softmax(torch.randn(k + 1, vocab) * 1.5, -1)
    q = torch.softmax(torch.randn(k, vocab) * 1.5, -1)
    gen = torch.Generator().manual_seed(1)
    drafts = torch.multinomial(q.repeat(trials, 1), 1, generator=gen).view(trials, k)
    tokens, num_acc = rejection_sample(
        p.expand(trials, -1, -1).contiguous(), q.expand(trials, -1, -1).contiguous(), drafts, gen
    )
    first = tokens[:, 0]
    empirical = torch.bincount(first, minlength=vocab).float() / trials
    torch.testing.assert_close(empirical, p[0], atol=4e-3, rtol=0)

    # Acceptance of the first draft token matches the analytic rate sum_x min(p, q).
    expected_accept = torch.minimum(p[0], q[0]).sum()
    assert float((num_acc > 0).float().mean()) == pytest.approx(float(expected_accept), abs=4e-3)


def test_rejection_sampler_identical_models_accept_everything():
    vocab, k = 10, 4
    p = torch.softmax(torch.randn(2, k + 1, vocab), -1)
    drafts = torch.multinomial(p[:, :k].reshape(-1, vocab), 1).view(2, k)
    tokens, num_acc = rejection_sample(p, p[:, :k].clone(), drafts)
    assert num_acc.tolist() == [k, k]
    assert (tokens[:, :k] == drafts).all() and (tokens[:, k] >= 0).all()


def test_rejection_sampler_greedy_one_hot():
    vocab = 5
    one_hot = lambda i: torch.nn.functional.one_hot(torch.tensor(i), vocab).float()  # noqa: E731
    p = torch.stack([one_hot(1), one_hot(2), one_hot(3)])[None]  # target wants 1, 2, then 3
    q = torch.stack([one_hot(1), one_hot(4)])[None]  # draft proposes 1, 4
    tokens, num_acc = rejection_sample(p, q, torch.tensor([[1, 4]]))
    assert num_acc.item() == 1
    assert tokens.tolist() == [[1, 2, -1]]  # accept 1, reject 4, correct to 2


@pytest.mark.parametrize("k", [1, 3, 5])
def test_greedy_speculative_output_equals_plain_greedy(tiny_model, k):
    prompts = [[1, 2, 3, 4, 5], [9, 8, 7], list(range(30, 52))]
    want = [dense_greedy(tiny_model, p, 12) for p in prompts]
    engine = InferenceEngine(
        make_config(tiny_model, speculative=SpeculativeConfig(num_speculative_tokens=k)),
        model=tiny_model,
    )
    outs = engine.generate(prompts, SamplingParams(max_tokens=12, temperature=0.0))
    assert [o.output_token_ids for o in outs] == want
    stats = engine.stats()["speculative"]
    assert stats["num_draft_tokens"] > 0
    assert 1.0 <= stats["mean_tokens_per_step"] <= k + 1


def test_perfect_draft_accepts_all_and_cuts_target_passes(tiny_model):
    """Draft == target: every proposal is accepted, k+1 tokens per verify pass."""
    k = 4
    draft = build_model(PRESETS["tiny"], seed=0)  # identical weights
    engine = InferenceEngine(
        make_config(tiny_model, speculative=SpeculativeConfig(num_speculative_tokens=k)),
        model=tiny_model, draft_model=draft,
    )
    out = engine.generate([[3, 1, 4, 1, 5]], SamplingParams(max_tokens=20, temperature=0.0))[0]
    assert out.output_token_ids == dense_greedy(tiny_model, [3, 1, 4, 1, 5], 20)
    stats = engine.stats()["speculative"]
    assert stats["acceptance_rate"] == pytest.approx(1.0)
    assert stats["per_token_acceptance"] == pytest.approx(1.0)
    # 1 prefill pass + ceil(19 / (k+1)) verify passes on the target.
    assert engine.spec.target.num_forward_passes == 1 + 4


def test_sampled_speculative_runs_and_respects_max_tokens(tiny_model):
    engine = InferenceEngine(
        make_config(tiny_model, speculative=SpeculativeConfig(num_speculative_tokens=3)),
        model=tiny_model,
    )
    params = SamplingParams(max_tokens=17, temperature=0.8, top_p=0.9, ignore_eos=True)
    outs = engine.generate([[1, 2, 3], [4, 5, 6, 7]], params)
    assert all(len(o.output_token_ids) == 17 for o in outs)


def test_layer_skip_draft_shares_parameters(tiny_model):
    draft = make_layer_skip_draft(tiny_model, 1)
    assert draft.num_layers == 1
    assert draft.embed_tokens is tiny_model.embed_tokens
    assert draft.layers[0] is tiny_model.layers[0]
    with pytest.raises(ValueError):
        make_layer_skip_draft(tiny_model, tiny_model.num_layers)
