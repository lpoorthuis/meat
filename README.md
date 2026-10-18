# meat

Abridge a code diff into a **reading diff**.

Humans need to review agent-written code in critical systems.
But models are good now. You don't need to review for style or nil-checks
or imports. You need to review concepts, algorithm choices, architecture.

So meat uses a model to reduce a diff to the important parts.
It shows you the meat.

Install from a local checkout:

```sh
go install ./cmd/meat
```

Run with `meat` to review the latest commit.
It takes git-looking parameters to pick commits to review.

By default meat uses OpenAI's `gpt-6.1-sol` with medium reasoning effort.

Copilot subscriptions and Google Cloud (Vertex AI) are supported too.
See `meat --help` for model selection and credentials.

It takes a while to process a commit for reading.
So I suggest you have an agent build `meat` into your devtools so that
it pre-processes it.

Very large diffs are split at file and hunk boundaries and abridged
chunk by chunk (up to a few MB), so one huge commit still produces a
single merged reading diff — it just takes proportionally longer.
