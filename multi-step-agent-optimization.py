# Using metrics and rewards to optimize multi step agents in DSPy
## Why functional workflows still fail users, and how to make behavior learnable using differentiable programming.

import re
FRIENDLY_WORDS = {"hi", "hey", "please", "thanks", "thank you", "sure", "happy"}

def friendly_eta_metric(ex, pred, trace=None):
    # 0. Tag must match
    if ex.tag != pred.tag:
        return 0.0

    # For non-ETA tickets, any correct tag scores 1.0
    if ex.tag != "eta":
        return 1.0

    body = pred.body.lower()

    rules = [
        "eta" in body,                           # mentions ETA
        any(w in body for w in FRIENDLY_WORDS),  # at least one polite word
        5 <= len(body.split()) <= 40,            # not too short / long
        re.search(r"\b\d{1,2}\s?min", body) is not None  # gives a minutes estimate
    ]

    return sum(rules) / len(rules)              # 0.0-1.0

import dspy
import re

# ── Signatures
class ServiceReply(dspy.Signature):
    tag:  str = dspy.OutputField()
    body: str = dspy.OutputField()

class RouterSignature(dspy.Signature):
    ticket: str = dspy.InputField()
    route:  str = dspy.OutputField(desc='eta | missing | driver | fallback')

class LatePathSignature(dspy.Signature):
    ticket: str = dspy.InputField()
    body:   str = dspy.OutputField(desc='Friendly ETA sentence that includes the word "eta".')

# ── Modules
class Router(dspy.Module):
    def __init__(self):
        self.step = dspy.Predict(RouterSignature)
    def forward(self, ticket: str):
        return self.step(ticket=ticket).route.lower().strip()

class LatePath(dspy.Module):
    def __init__(self):
        self.step = dspy.Predict(LatePathSignature)
    def forward(self, ticket: str):
        body = self.step(ticket=ticket).body.strip()
        return dspy.Prediction(tag="eta", body=body, _sig=ServiceReply)

class MissingPath(dspy.Module):
    def forward(self, ticket: str):
        return dspy.Prediction(tag="missing",
                               body="Item verified missing via photo. Refund has been issued.",
                               _sig=ServiceReply)

class DriverPath(dspy.Module):
    def forward(self, ticket: str):
        return dspy.Prediction(tag="driver",
                               body="Driver located, contact info sent.",
                               _sig=ServiceReply)

class FallbackPath(dspy.Module):
    def forward(self, ticket: str):
        return dspy.Prediction(tag="fallback",
                               body="Please see our FAQ or reach live support.",
                               _sig=ServiceReply)

# ── Top-level agent
class SupportAgent(dspy.Module):
    def __init__(self):
        self.router   = Router()
        self.eta      = LatePath()
        self.missing  = MissingPath()
        self.driver   = DriverPath()
        self.fallback = FallbackPath()
    def forward(self, ticket: str):
        route = self.router(ticket=ticket)
        if route == "eta":     return self.eta(ticket=ticket)
        if route == "missing": return self.missing(ticket=ticket)
        if route == "driver":  return self.driver(ticket=ticket)
        return self.fallback(ticket=ticket)


THREADS = 1
evaluate = dspy.Evaluate(
    devset=dev,
    metric=friendly_eta_metric,
    num_threads=THREADS,
    display_progress=True,
    display_table=5,
)

support_bot = supportAgent()
evaluate(support_bot)

teacher_lm = dspy.LM('openai/gpt-4o')

THREADS=1
optimizer = dspy.MIPROv2(
    metric=friendly_eta_metric,
    auto="light",                 # minimal search space
    num_threads=THREADS,
    teacher_settings=dict(lm=teacher_lm),
    prompt_model=dspy.settings.lm # reuse default mini LM for prompts
)

optimized_support_bot = optimizer.compile(
    SupportAgent(),               # program to optimize
    trainset=train[:100],
    requires_permission_to_run=False,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
)

evaluate(optimized_support_bot)


