"""
Wikipedia Environment — faithful to the paper's tool design.

From the paper (Section 3):
    "We design a simple Wikipedia web API with three actions:
     (1) search[entity] — returns the first 5 sentences of the entity wiki page
         if it exists, or suggests top-5 similar entities
     (2) lookup[string] — returns the next sentence containing string in the
         current page
     (3) finish[answer] — finishes the task with answer"

This is NOT a generic tool framework. The paper's insight is that even a minimal
set of grounded actions (search + lookup) dramatically reduces hallucination
compared to pure reasoning (CoT), because the model can VERIFY its beliefs
by actually looking things up.

Why Wikipedia specifically?
    - It's the knowledge source behind most QA benchmarks (HotpotQA, FEVER)
    - It provides the "external grounding" that pure CoT lacks
    - The search→lookup pattern mimics how humans browse: find a page, then scan for details
"""

import requests
import re


class WikipediaEnv:
    """
    Simulates the paper's Wikipedia environment.

    Maintains state across steps — lookup depends on the page found by search.
    This statefulness is important: it means the agent must plan its actions
    in sequence (search first, then lookup), which is part of what ReAct learns.
    """

    # Wikipedia API requires a User-Agent header, otherwise it may return
    # HTML error pages instead of JSON, causing parse failures.
    HEADERS = {"User-Agent": "ReActAgent/1.0 (educational project)"}

    def __init__(self):
        self.current_page = None      # Full text of the current Wikipedia page
        self.current_paragraphs = []  # Paragraphs for lookup
        self.lookup_index = 0         # Track position for successive lookups
        self.lookup_keyword = None    # Current keyword being looked up

    def search(self, entity: str) -> str:
        """
        Search Wikipedia for an entity.
        Returns the first few sentences if found, or similar entity suggestions.

        This maps directly to the paper's search[entity] action.
        """
        entity = entity.strip()

        # Use Wikipedia API to search
        try:
            # First, try to get the page directly
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "titles": entity,
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "redirects": 1,
                "format": "json",
            }
            response = requests.get(url, params=params, headers=self.HEADERS, timeout=10)
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            page = next(iter(pages.values()))

            if "missing" in page:
                # Page not found — search for similar entities
                return self._search_similar(entity)

            extract = page.get("extract", "")
            if not extract:
                return self._search_similar(entity)

            # Store for lookup
            self.current_page = extract
            self.current_paragraphs = [p.strip() for p in extract.split("\n") if p.strip()]
            self.lookup_index = 0
            self.lookup_keyword = None

            # Return first 5 sentences (matching the paper)
            sentences = self._split_sentences(extract)
            result = " ".join(sentences[:5])

            return result if result else f"Could not find [{entity}]. Similar: {self._search_similar(entity)}"

        except Exception as e:
            return f"Search error: {str(e)}"

    def lookup(self, keyword: str) -> str:
        """
        Look up a keyword in the current page.
        Returns the next sentence containing the keyword.

        This maps to the paper's lookup[string] action.
        The sequential nature is key — if the agent calls lookup twice
        with the same keyword, it gets the NEXT occurrence, not the same one.
        """
        if self.current_page is None:
            return "(Result 1 / 1) No page loaded. Use search first."

        keyword = keyword.strip().lower()

        # Reset index if keyword changed
        if keyword != self.lookup_keyword:
            self.lookup_keyword = keyword
            self.lookup_index = 0

        sentences = self._split_sentences(self.current_page)
        matching = [(i, s) for i, s in enumerate(sentences) if keyword in s.lower()]

        if not matching:
            return f"(Result 0 / 0) The keyword '{keyword}' was not found on this page."

        # Find next match from current position
        for idx, (sent_idx, sentence) in enumerate(matching):
            if sent_idx >= self.lookup_index:
                self.lookup_index = sent_idx + 1
                return f"(Result {idx + 1} / {len(matching)}) {sentence}"

        # Wrapped around
        self.lookup_index = matching[0][0] + 1
        return f"(Result 1 / {len(matching)}) {matching[0][1]}"

    def _search_similar(self, entity: str) -> str:
        """Search for similar entities when exact match fails."""
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "opensearch",
                "search": entity,
                "limit": 5,
                "format": "json",
            }
            response = requests.get(url, params=params, headers=self.HEADERS, timeout=10)
            data = response.json()

            if len(data) > 1 and data[1]:
                suggestions = data[1][:5]
                return f"Could not find [{entity}]. Similar: {suggestions}"
            return f"Could not find [{entity}]. No similar entities found."

        except Exception:
            return f"Could not find [{entity}]."

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences. Simple but effective."""
        # Split on period followed by space or end of string
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def step(self, action: str, action_input: str) -> str:
        """
        Execute an action and return the observation.
        This is the environment interface that the agent calls.
        """
        action = action.strip().lower()

        if action == "search":
            return self.search(action_input)
        elif action == "lookup":
            return self.lookup(action_input)
        elif action == "finish":
            return action_input  # The answer itself
        else:
            return f"Unknown action: {action}. Valid actions: search, lookup, finish."

    def reset(self):
        """Reset environment state between questions."""
        self.current_page = None
        self.current_paragraphs = []
        self.lookup_index = 0
        self.lookup_keyword = None
