# Code for 
# Differentiable Programming for Learnable Graphs: Optimizing Agent Workflows with DSPy
import dspy
# Configure your dspy LM appropriately

class ServiceReply(dspy.Signature):
    tag:  str = dspy.OutputField()   # eta | missing | driver | fallback
    body: str = dspy.OutputField()   # free-form message

# ── Router with closed label set 
# constrained‐label LLM node that inspects the incoming support ticket and returns one of four route tags: eta, missing, driver, or fallback.

class RouterSignature(dspy.Signature):
    """Choose the single best route for this customer ticket."""
    ticket: str = dspy.InputField()
    route: str = dspy.OutputField(desc='Choose the single best route for this customer ticket from: eta | missing | driver | fallback.')

class Router(dspy.Module):
    def __init__(self):
        self.step = dspy.Predict(RouterSignature) # Use the signature class directly
    def forward(self, ticket:str):
        return self.step(ticket=ticket).route.lower().strip()

# ── Branch modules (use unified ServiceReply) 
# synthesises a plausible delivery window + turns that timestamp into a customer-friendly sentence
# returns ServiceReply

class LatePath(dspy.Module):
    def forward(self, ticket: str):
        mins = random.randint(10, 20)
        eta  = (dt.datetime.now() + dt.timedelta(minutes=mins)).strftime("%I:%M %p")
        msg  = f"Courier is about {mins} min away — arriving ≈ {eta}."
        return dspy.Prediction(tag="eta", body=msg, _sig=ServiceReply)

# confirms the missing-item claim and issues a refund message
class MissingPath(dspy.Module):
    def forward(self, ticket: str):
        msg = "Item verified missing via photo. Refund has been issued."
        return dspy.Prediction(tag="missing", body=msg, _sig=ServiceReply)

# a stub that would locate the courier and respond with tag="driver"
class DriverPath(dspy.Module):
    def forward(self, ticket:str):
        return dspy.Prediction(tag="driver",
                               body="Driver located, contact info sent.",
                               _sig=ServiceReply)

# a final catch-all that directs the user to FAQ or human support, returning tag="fallback"
class FallbackPath(dspy.Module):
    def forward(self, ticket:str):
        return dspy.Prediction(tag="fallback",
                               body="Please see our FAQ or reach live support.",
                               _sig=ServiceReply)

# ── Top-level agent 
class SupportAgent(dspy.Module):
    def __init__(self):
        self.router  = Router()
        self.eta     = LatePath()
        self.missing = MissingPath()
        self.driver  = DriverPath()
        self.fallback= FallbackPath()
    def forward(self, ticket:str):
        route = self.router(ticket=ticket)
        if route == "eta":      return self.eta(ticket=ticket)
        if route == "missing":  return self.missing(ticket=ticket)
        if route == "driver":   return self.driver(ticket=ticket)
        return self.fallback(ticket=ticket)

supportbot = SupportAgent()
supportbot(ticket="Order #8123 is 20 minutes late. Any ETA?")

"""
Prediction(
    tag='eta',
    body='Courier is about 12 min away — arriving ≈ 08:55 PM.',
    _sig=ServiceReply( -> tag, body
    instructions='Given the fields , produce the fields `tag`, `body`.'
    tag = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Tag:', 'desc': '${tag}'})
    body = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Body:', 'desc': '${body}'})
)
)
"""

# ── Tiny dataset

train = [
    Example(ticket="Order #8123 is 20 minutes late. Any ETA?",         tag="eta").with_inputs("ticket"),
    Example(ticket="Driver stuck in traffic for order #9041.",        tag="driver").with_inputs("ticket"),
    Example(ticket="Burger arrived but fries missing in order #6677.", tag="missing").with_inputs("ticket"),
    Example(ticket="App shows delivered yet nothing here (#7001).",    tag="missing").with_inputs("ticket"),
    Example(ticket="Order #5502 delayed, need update please.",         tag="eta").with_inputs("ticket"),
    Example(ticket="I didn't get the soda in combo meal #5502.",       tag="missing").with_inputs("ticket"),
    Example(ticket="Driver phone off — can you locate? Order #4321.",  tag="driver").with_inputs("ticket"),
    Example(ticket="How do I cancel my late order #8890?",             tag="fallback").with_inputs("ticket"),
    Example(ticket="Missing dipping sauce order #9988.",               tag="missing").with_inputs("ticket"),
    Example(ticket="Order #7002 already 25 min late. Where is it?",     tag="eta").with_inputs("ticket"),
]

dev = [
    Example(ticket="Order #1234 is late — where is it?",               tag="eta").with_inputs("ticket"),
    Example(ticket="Missing fries in order #5678, need refund.",       tag="missing").with_inputs("ticket"),
    Example(ticket="Driver got flat tire. What's new ETA for #2020?",  tag="eta").with_inputs("ticket"),
    Example(ticket="Never got my drink with order #3003.",             tag="missing").with_inputs("ticket"),
    Example(ticket="Order #4040 taking forever. Any update?",          tag="eta").with_inputs("ticket"),
    Example(ticket="Is there a way to track courier? Order #5050.",    tag="driver").with_inputs("ticket"),
    Example(ticket="Half my toppings missing on pizza #6060.",         tag="missing").with_inputs("ticket"),
    Example(ticket="App shows delivered but nothing arrived (#7070).", tag="missing").with_inputs("ticket"),
    Example(ticket="FAQ didn't help. Cancel late order #8080.",        tag="fallback").with_inputs("ticket"),
    Example(ticket="What's status of order #9090? It's 30 min late.",  tag="eta").with_inputs("ticket"),
]

# ── Simple metric function that compares predicted tags only 
def tag_match(ex, pred, trace=None):
    return float(ex.tag == pred.tag)

# ── Baseline eval
eval_dev = dspy.Evaluate(devset=dev, metric=tag_match, num_threads=1,
                    display_progress=True, display_table=5)
print("Zero-shot dev score:", eval_dev(SupportAgent()))


# simply create a DSPy optimizer that uses the training set and our LM
# and iterates through to find better prompts at each step
opt = dspy.MIPROv2(metric=tag_match, auto="light", num_threads=1)
agent_optim = opt.compile(SupportAgent(), trainset=train,
                          requires_permission_to_run=False,
                          max_bootstrapped_demos=2, max_labeled_demos=4)

print("Post-opt dev score:", eval_dev(agent_optim))

# output
"""
2025/07/13 20:38:37 INFO dspy.teleprompt.mipro_optimizer_v2: 
RUNNING WITH THE FOLLOWING LIGHT AUTO RUN SETTINGS:
num_trials: 10
minibatch: False
num_fewshot_candidates: 6
num_instruct_candidates: 3
valset size: 8

2025/07/13 20:38:37 INFO dspy.teleprompt.mipro_optimizer_v2: 
==> STEP 1: BOOTSTRAP FEWSHOT EXAMPLES <==
2025/07/13 20:38:37 INFO dspy.teleprompt.mipro_optimizer_v2: These will be used as few-shot example candidates for our program and for creating instructions.

2025/07/13 20:38:37 INFO dspy.teleprompt.mipro_optimizer_v2: Bootstrapping N=6 sets of demonstrations...
Bootstrapping set 1/6
Bootstrapping set 2/6
....
Bootstrapping set 6/6
 50%|█████     | 1/2 [00:00<00:00, 362.92it/s]
2025/07/13 20:38:40 INFO dspy.teleprompt.mipro_optimizer_v2: 
==> STEP 2: PROPOSE INSTRUCTION CANDIDATES <==
2025/07/13 20:38:40 INFO dspy.teleprompt.mipro_optimizer_v2: We will use the few-shot examples from the previous step, a generated dataset summary, a summary of the program code, and a randomly selected prompting tip to propose instructions.
Bootstrapped 1 full traces after 1 examples for up to 1 rounds, amounting to 1 attempts.
2025/07/13 20:38:42 INFO dspy.teleprompt.mipro_optimizer_v2:
2025/07/13 20:39:05 INFO dspy.teleprompt.mipro_optimizer_v2: Proposed Instructions for Predictor 0:
2025/07/13 20:39:05 INFO dspy.teleprompt.mipro_optimizer_v2: 0: Choose the single best route for this customer ticket.
2025/07/13 20:39:05 INFO dspy.teleprompt.mipro_optimizer_v2: 1: You are a customer support routing assistant for a delivery and logistics business. Your task is to analyze the content of a customer ticket and classify it into one of the following categories: `eta` (estimated time of arrival), `missing` (missing item issues), `driver` (driver-related concerns), or `fallback` (general inquiries that do not fit the other categories). Carefully read the ticket and choose the single best route that matches the customer's concern. Respond with the selected category.

2025/07/13 20:39:05 INFO dspy.teleprompt.mipro_optimizer_v2: 2: Analyze the content of the provided customer ticket and classify it into one of the following categories that best describes the issue: `eta` (inquiries about estimated time of arrival), `missing` (reports of missing items), `driver` (issues related to the driver or delivery personnel), or `fallback` (general or unsupported issues). Return only the category name that most accurately matches the ticket's context.

2025/07/13 20:39:05 INFO dspy.teleprompt.mipro_optimizer_v2: 

2025/07/13 20:39:05 INFO dspy.teleprompt.mipro_optimizer_v2: ==> STEP 3: FINDING OPTIMAL PROMPT PARAMETERS <==
2025/07/13 20:39:05 INFO dspy.teleprompt.mipro_optimizer_v2: We will evaluate the program over a series of trials with different combinations of instructions and few-shot examples to find the optimal combination using Bayesian Optimization.

2025/07/13 20:39:05 INFO dspy.teleprompt.mipro_optimizer_v2: == Trial 1 / 10 - Full Evaluation of Default Program ==
Average Metric: 7.00 / 8 (87.5%): 100%|██████████| 8/8 [00:05<00:00,  1.47it/s]2025/07/13 20:39:11 INFO dspy.evaluate.evaluate: Average Metric: 7.0 / 8 (87.5%)
2025/07/13 20:39:11 INFO dspy.teleprompt.mipro_optimizer_v2: Default program score: 87.5
...



