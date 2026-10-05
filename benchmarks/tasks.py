import random
import re


def score_answer(answer, references):
    """Score against ground truth, never against baseline-generated text."""
    hits = [
        bool(
            re.search(
                r"(?<![\w-])" + re.escape(ref) + r"(?![\w-])",
                answer,
                flags=re.I,
            )
        )
        for ref in references
    ]
    return {
        "score": sum(hits) / len(hits),
        "correct": hits,
        "all_correct": all(hits),
    }


def chat_parts(tokenizer, question):
    marker = "LOWRAM_DOCUMENT_INSERTION_POINT"
    content = f"Read the document and answer using only its facts.\n\n{marker}\n\n{question}"
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = content + "\nAnswer:"
    start, end = text.split(marker)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    return encode(start), encode(end)


def retrieval_examples(tokenizer, context_tokens, samples, seed):
    """Exact-length prompts with three facts distributed through the document."""
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    question = (
        "What are the access codes for Atlas, Boreal, and Cygnus? "
        "Return only the three codes, in that order, separated by commas."
    )
    prefix, suffix = chat_parts(tokenizer, question)
    examples = []
    for index in range(samples):
        rng = random.Random(seed + index)
        codes = [
            f"{rng.randrange(1000, 10000)}-{rng.randrange(1000, 10000)}"
            for _ in range(3)
        ]
        names = ["Atlas", "Boreal", "Cygnus"]
        facts = [
            f"\nThe access code for project {name} is {code}.\n"
            for name, code in zip(names, codes)
        ]
        fact_ids = [encode(fact) for fact in facts]
        filler = encode(
            "\n".join(
                f"Archive entry {i}: team {rng.choice(['Amber', 'Cedar', 'Delta', 'Elm'])} "
                f"reviewed {rng.randrange(10, 90)} routine records in sector {rng.randrange(100, 999)}. "
                "The maintenance log was checked and the daily report was filed."
                for i in range(120)
            )
        )
        budget = (
            context_tokens - len(prefix) - len(suffix) - sum(map(len, fact_ids))
        )
        if budget < 64:
            raise ValueError(
                "Context is too short for the retrieval question and facts."
            )
        filler = (filler * (budget // len(filler) + 1))[:budget]
        body, positions, last = [], [], 0
        for fraction, ids in zip((0.15, 0.5, 0.85), fact_ids):
            boundary = int(budget * fraction)
            body.extend(filler[last:boundary])
            positions.append((len(prefix) + len(body)) / context_tokens)
            body.extend(ids)
            last = boundary
        body.extend(filler[last:])
        ids = prefix + body + suffix
        assert len(ids) == context_tokens
        examples.append(
            {
                "id": f"retrieval-{seed}-{index}",
                "question": question,
                "document": tokenizer.decode(body),
                "input_ids": ids,
                "references": codes,
                "fact_labels": names,
                "facts": facts,
                "fact_positions": positions,
                "metric": "fact_recall",
            }
        )
    return examples

