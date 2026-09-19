"""Standalone DSPy reference: native Predict -> MIPROv2.compile -> native prediction."""

import dspy
from dspy.utils.callback import BaseCallback

from examples.sms.common import PROBLEM
from examples.sms.common import NativeRun
from examples.sms.common import optimizer_model
from examples.sms.common import setup
from examples.sms.common import target_model
from prompt_optimiser.metrics import exact_match
from prompt_optimiser.vllm import VLLM


def main():
    args, splits, metadata = setup("native-dspy")
    train, validation, _test = splits
    target = target_model(args)
    proposal = optimizer_model(args)
    lm = target.as_dspy() if isinstance(target, VLLM) else dspy.LM(target)
    proposer = (
        proposal.as_dspy() if isinstance(proposal, VLLM) else dspy.LM(proposal) if proposal else lm
    )
    signature = dspy.Signature("text -> answer", instructions=args.seed_prompt or PROBLEM)
    initial = dspy.Predict(signature)
    trainset = [dspy.Example(text=r.input, answer=r.target).with_inputs("text") for r in train]
    valset = [dspy.Example(text=r.input, answer=r.target).with_inputs("text") for r in validation]

    def metric(example, prediction, trace=None):
        return exact_match(example.answer, prediction.answer)

    with NativeRun(args, metadata) as run:

        class Progress(BaseCallback):
            def __init__(self):
                self.prompts = {}
                self.error = None

            def on_evaluate_start(self, call_id, instance, inputs):
                self.prompts[call_id] = inputs["program"].signature.instructions

            def on_evaluate_end(self, call_id, outputs, exception=None):
                prompt = self.prompts.pop(call_id)
                if exception is None:
                    try:
                        run.progress(prompt, outputs.score / 100)
                    except Exception as exc:
                        self.error = exc

        progress = Progress()
        optimizer = dspy.MIPROv2(
            metric=metric,
            task_model=lm,
            prompt_model=proposer,
            auto=None,
            num_candidates=args.num_candidates,
            max_labeled_demos=args.max_demos,
            max_bootstrapped_demos=0,
            seed=args.seed,
            num_threads=1,
            max_errors=1,
        )
        with dspy.context(lm=lm, callbacks=[progress]):
            best = optimizer.compile(
                initial,
                trainset=trainset,
                valset=valset,
                num_trials=args.steps,
                minibatch=False,
                program_aware_proposer=False,
                data_aware_proposer=True,
                view_data_batch_size=len(trainset),
                tip_aware_proposer=False,
            )
        if progress.error is not None:
            raise progress.error
        initial.save(args.output / "baseline-program.json")
        best.save(args.output / "best-program.json")
        # The held-out test set is first evaluated here, after compile has selected the winner.
        with dspy.context(lm=lm):
            run.finish(
                lambda text: initial(text=text).answer,
                lambda text: best(text=text).answer,
                splits,
                best.signature.instructions,
                exact_match,
            )


if __name__ == "__main__":
    main()
