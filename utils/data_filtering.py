def filter_tasks_by_min_seq_len(task_dict, tokenizer, min_seq_len):
    """Wrap each task's process_docs so only docs whose rendered prompt
    tokenises to at least min_seq_len tokens are evaluated."""

    def make_process_docs(task, existing):
        def process_docs(dataset):
            if existing is not None:
                dataset = existing(dataset)
            num_before = len(dataset)
            dataset = dataset.filter(
                lambda doc: len(
                    tokenizer(
                        task.doc_to_text(doc), add_special_tokens=False
                    ).input_ids
                )
                >= min_seq_len
            )
            print(
                f"{task.config.task}: kept {len(dataset)}/{num_before} docs "
                f"with >= {min_seq_len} prompt tokens"
            )
            return dataset

        return process_docs

    for _, task in task_dict.items():
        task.config.process_docs = make_process_docs(
            task, task.config.process_docs
        )
    return list(task_dict.values())