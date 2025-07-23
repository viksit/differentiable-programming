
# tool controller
import torch, torch.nn as nn
import pandas as pd, random, statistics, time, dspy
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNNRouter(nn.Module):
    def __init__(self, vocab, dim=64):
        super().__init__()
        self.vocab = vocab
        self.emb   = nn.Embedding(4096, dim, padding_idx=0)
        self.rnn   = nn.GRU(dim, dim, batch_first=True)
        self.lin   = nn.Linear(dim, 2)
    def _tok(self, txt):
        ids=[self.vocab.setdefault(w, len(self.vocab)+1)
             for w in txt.lower().split()]
        return torch.tensor([ids], device=device)
    def forward(self, txt):
        x,_=self.rnn(self.emb(self._tok(txt))); return self.lin(x[:,-1])

vocab, rnn = {}, RNNRouter(vocab).to(device)
opt, loss_fn = torch.optim.Adam(rnn.parameters(), 1e-3), nn.CrossEntropyLoss()

for _ in range(3):
    for row in df_train.itertuples():
        y=torch.tensor([0] if row.label=="REFUND" else [1], device=device)
        loss=loss_fn(rnn(row.ticket_text), y)
        opt.zero_grad(); loss.backward(); opt.step()



# --- sanity check --------------------------------------------------
import torch.nn.functional as F
from random import sample

def softmax_conf(logits):
    probs = F.softmax(logits, dim=-1)[0]
    return float(probs.max()), "REFUND" if probs.argmax()==0 else "NO_REFUND"

test_snip = sample(list(df_test.itertuples()), k=10)   # 10 random tickets
hits = 0

for row in test_snip:
    with torch.no_grad():
        conf, pred = softmax_conf(rnn(row.ticket_text))
    correct = "✓" if pred == row.label else "✗"
    hits += (pred == row.label)
    print(f"{correct}  {pred:<8}  conf={conf:.2f} | {row.ticket_text[:60]}...")

print(f"\nMini-accuracy: {hits}/{len(test_snip)} = {hits/len(test_snip):.1%}")

## Output
✓  REFUND    conf=1.00 | My order #7892 arrived with the wrong color sneakers. I orde...
✓  REFUND    conf=1.00 | I returned the dress 3 weeks ago per your return policy, but...
...

# dspy routing and planning
class PlanSig(dspy.Signature):
    """Return exactly REFUND or NO_REFUND."""
    ticket: str = dspy.InputField()
    label : str = dspy.OutputField(desc="REFUND or NO_REFUND")

class ReplySig(dspy.Signature):
    """Write a one-sentence customer-support reply."""
    ticket: str = dspy.InputField()
    outcome: str = dspy.InputField(desc="REFUND or NO_REFUND")
    reply : str = dspy.OutputField()

class GPTRouter(dspy.Module):
    def forward(self, ticket):
        label = dspy.Predict(PlanSig)(ticket=ticket).label.strip().upper()
        if "REFUND" not in label: label = "NO_REFUND"
        reply = dspy.Predict(ReplySig)(ticket=ticket, outcome=label).reply
        return dspy.Prediction(label=label, reply=reply)

class RNNPlusGPT(dspy.Module):
    def forward(self, ticket):
        label = "REFUND" if rnn(ticket).argmax().item()==0 else "NO_REFUND"
        reply = dspy.Predict(ReplySig)(ticket=ticket, outcome=label).reply
        return dspy.Prediction(label=label, reply=reply)

devset = [
    dspy.Example(ticket=row.ticket_text, label=row.label).with_inputs("ticket")
    for row in df_test.itertuples()
]

# accuracy eval
def accuracy_metric(ex, pred, _=None): return float(ex.label == pred.label)

eval_fn = dspy.Evaluate(metric=accuracy_metric, devset=devset, display_progress=True)

print("GPT-only agent accuracy:", eval_fn(GPTRouter()))
print("RNN + GPT agent accuracy:", eval_fn(RNNPlusGPT()))

def eval_cost(agent, name):
    # clear LM call history
    lm.history = []

    # accuracy + latency
    times, good = [], 0
    for row in df_test.itertuples():
        t0 = time.time()
        pred = agent(ticket=row.ticket_text).label
        times.append(time.time() - t0)
        good += (pred == row.label)
    acc = good / len(df_test)
    lat = statistics.median(times) * 1e3

    # $$$ from LiteLLM cost tracking
    cost = sum(x["cost"] for x in lm.history if x.get("cost") is not None)

    return dict(router=name, acc=acc, ms=lat, usd=cost)

gpt_stats = eval_cost(GPTRouter(),  "GPT-only")
rnn_stats = eval_cost(RNNPlusGPT(), "RNN + GPT")

print(
    tabulate(
        [list(gpt_stats.values()), list(rnn_stats.values())],
        headers=list(gpt_stats.keys()),
        floatfmt=".3f",
    )
)

## Output
router       acc     ms    usd
---------  -----  -----  -----
GPT-only   1.000  0.389  0.479
RNN + GPT  1.000  0.690  0.289
