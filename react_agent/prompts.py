"""
Few-shot prompts for ReAct — the most important part of the paper.

From the paper (Section 3.1):
    "For tasks where reasoning is of primary importance, we alternate the
     generation of domain-specific thoughts and environment-specific actions
     so that the task-solving trajectory consists of multiple thought-action-
     observation steps."

The few-shot examples are human-written demonstrations of GOOD reasoning.
They teach the model:
    1. WHEN to think (before acting, after observing)
    2. WHAT to think about (decompose the question, interpret observations)
    3. HOW to act (which tool to use, what input to give)
    4. WHEN to stop (confident enough to call finish)

Key insight: The thought traces are NOT just chain-of-thought.
They serve specific purposes:
    - Decomposing complex questions into sub-questions
    - Extracting relevant info from observations
    - Tracking what's been found vs. what's still needed
    - Making judgments about when enough evidence exists

These exemplars are adapted from the paper's appendix and the official
GitHub repo (ysymyth/ReAct).
"""

# ─────────────────────────────────────────────────────────
# HotpotQA — Multi-hop Question Answering
# ─────────────────────────────────────────────────────────
# The model must answer questions that require reasoning across
# multiple Wikipedia articles (e.g., "Were X and Y born in the same city?")

HOTPOTQA_EXAMPLES = """Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector extends into, then find the elevation range of that area.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas. This orogeny is responsible for the formation of the Rocky Mountains. The eastern sector of the orogeny extended into the High Plains.
Thought 2: The eastern sector extends into the High Plains. I need to search High Plains and find its elevation range.
Action 2: Search[High Plains]
Observation 2: High Plains refers to one of the two distinct land regions of the Great Plains of the United States. The High Plains are a subregion of the Great Plains, mostly above 1800 feet, with some areas reaching over 7000 feet. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).
Thought 3: The elevation range of the High Plains is approximately 1,800 to 7,000 ft. I have enough information to answer.
Action 3: Finish[approximately 1,800 to 7,000 ft]

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who was named after who?
Thought 1: I need to find out who Milhouse from The Simpsons was named after.
Action 1: Search[Milhouse Van Houten]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening. Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.
Thought 2: Milhouse was named after Richard Nixon. The answer is Richard Nixon.
Action 2: Finish[Richard Nixon]

Question: Which documentary is about Finnish rock groups, 'The Saimaa Gesture' or 'Global Metal'?
Thought 1: I need to search for both documentaries and see which one is about Finnish rock groups.
Action 1: Search[The Saimaa Gesture]
Observation 1: The Saimaa Gesture (Finnish: Saimaa-ilmiö) is a 1981 Finnish documentary film directed by Aki and Mika Kaurismäki. It follows three Finnish rock groups — Juice Leskinen, Eppu Normaali, and Hassisen Kone — on a tour around Lake Saimaa.
Thought 2: The Saimaa Gesture is about Finnish rock groups. I don't even need to search Global Metal to confirm.
Action 2: Finish[The Saimaa Gesture]"""


HOTPOTQA_PROMPT = """Solve a question answering task with interleaving Thought, Action, Observation steps.

Thought can reason about the current situation.
Action can be three types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last successful search result.
(3) Finish[answer], which returns the answer and finishes the task.

You may take as many steps as are necessary.

Here are some examples:
{examples}

Question: {question}
{trajectory}"""


# ─────────────────────────────────────────────────────────
# FEVER — Fact Verification
# ─────────────────────────────────────────────────────────
# The model must classify claims as SUPPORTS, REFUTES, or NOT ENOUGH INFO
# by searching for evidence. This tests the model's ability to gather
# evidence and make judgments — not just retrieve answers.

FEVER_EXAMPLES = """Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
Thought 1: I need to search Nikolaj Coster-Waldau and find if he worked with Fox Broadcasting Company.
Action 1: Search[Nikolaj Coster-Waldau]
Observation 1: Nikolaj William Coster-Waldau (born 27 July 1970) is a Danish actor and producer. He graduated from the Danish National School of Performing Arts in Copenhagen in 1993. He is best known for his role as Jaime Lannister in the HBO series Game of Thrones. He also appeared in the Fox television show New Amsterdam.
Thought 2: He appeared in "New Amsterdam" on Fox. The claim says he worked with Fox Broadcasting Company, and since New Amsterdam aired on Fox, this supports the claim.
Action 2: Finish[SUPPORTS]

Claim: Stranger Things is set in Bloomington, Indiana.
Thought 1: I should search for Stranger Things to verify where it is set.
Action 1: Search[Stranger Things]
Observation 1: Stranger Things is an American science fiction horror drama television series created by the Duffer Brothers. Set in the 1980s, primarily in the fictional town of Hawkins, Indiana, the series centers on a number of mysteries and supernatural events.
Thought 2: Stranger Things is set in Hawkins, Indiana, not Bloomington, Indiana. This refutes the claim.
Action 2: Finish[REFUTES]

Claim: "Beautiful" by Christina Aguilera reached number two on the Billboard Hot 100.
Thought 1: I need to search the song "Beautiful" by Christina Aguilera and check its chart performance.
Action 1: Search[Beautiful Christina Aguilera song]
Observation 1: "Beautiful" is a song recorded by American singer Christina Aguilera for her fourth studio album, Stripped (2002). The song was written and produced by Linda Perry. It was released as the second single from the album. The song reached number two on the Billboard Hot 100.
Thought 2: The observation confirms the song reached number two on the Billboard Hot 100. This supports the claim.
Action 2: Finish[SUPPORTS]"""


FEVER_PROMPT = """Determine whether a claim is true (SUPPORTS), false (REFUTES), or if there is NOT ENOUGH INFO. Use interleaving Thought, Action, Observation steps.

Thought can reason about the current situation.
Action can be three types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last successful search result.
(3) Finish[answer], which returns the answer (SUPPORTS, REFUTES, or NOT ENOUGH INFO) and finishes the task.

Here are some examples:
{examples}

Claim: {question}
{trajectory}"""


def build_prompt(task: str, question: str, trajectory: str = "") -> str:
    """
    Build the full prompt for a given task and question.

    The trajectory parameter accumulates the agent's reasoning so far —
    each Thought/Action/Observation gets appended, building context.
    This is how the model "remembers" what it has done.
    """
    if task == "hotpotqa":
        return HOTPOTQA_PROMPT.format(
            examples=HOTPOTQA_EXAMPLES,
            question=question,
            trajectory=trajectory,
        )
    elif task == "fever":
        return FEVER_PROMPT.format(
            examples=FEVER_EXAMPLES,
            question=question,
            trajectory=trajectory,
        )
    else:
        raise ValueError(f"Unknown task: {task}. Choose 'hotpotqa' or 'fever'.")
