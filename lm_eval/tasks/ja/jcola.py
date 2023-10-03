"""
JCoLA: Japanese Corpus of Linguistic Acceptability
https://arxiv.org/pdf/2309.12676.pdf

JCoLA is a novel dataset for targeted syntactic evaluations of language models in Japanese, which consists of 10,020 sentences with acceptability judgments by linguists. The sentences are manually extracted from linguistics journals, handbooks and textbooks.

Homepage: https://github.com/osekilab/JCoLA/tree/main
"""
from lm_eval.tasks.glue import CoLA
from lm_eval.base import rf

_CITATION = """
@article{someya2023jcola,
      title={JCoLA: Japanese Corpus of Linguistic Acceptability}, 
      author={Taiga Someya and Yushi Sugimoto and Yohei Oseki},
      year={2023},
      eprint={2309.12676},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

class JCoLA(CoLA):
    VERSION = 0.1
    PROMPT_VERSION = 0.1
    DATASET_PATH = "shunk031/JGLUE"
    DATASET_NAME = "JCoLA"
    SEP = "\n"
    # 1: acceptable, 0: unacceptable
    CHOICES = {1: "はい", 0: "いいえ"}

    def doc_to_text(self, doc):
        # "{}\nQuestion: Does this sentence make sense?\nAnswer:"
        return "{}{}質問: この文は文法的ですか？{}答え:".format(
            doc["sentence"], self.SEP, self.SEP
        )

    def doc_to_target(self, doc):
        return " {}".format(self.CHOICES[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " %s" % self.CHOICES[1])
        ll_false, _ = rf.loglikelihood(ctx, " %s" % self.CHOICES[0])
        return ll_true, ll_false


class JCoLAWithJAAlpacaPrompt(JCoLA):
    """
    Reference:
    - data: https://huggingface.co/datasets/fujiki/japanese_alpaca_data
    - code: https://github.com/Stability-AI/gpt-neox/blob/c130a4edc1120dccec8f02a34eb60d3e8f484cd3/finetune/finetune_base_ja.py#LL118C23-L127C11
    """
    PROMPT_VERSION = 0.3
    DESCRIPTION = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n"
    INSTRUCTION = f"与えられた文が文法的であるかを回答してください。\n\n出力は以下から選択してください：\n" + "\n".join(list(JCoLA.CHOICES.values()))

    def doc_to_text(self, doc):
        """
        以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

        ### 指示: 
        {instruction}

        ### 入力: 
        {input}

        ### 応答: 
        {response}
        """
        input_text = doc["sentence"]
        return f"### 指示:\n{self.INSTRUCTION}\n\n### 入力:\n{input_text}\n\n### 応答:\n"


class JCoLAWithRinnaInstructionSFT(JCoLA):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft
    """
    PROMPT_VERSION = 0.4
    DESCRIPTION = "ユーザー: " + f"与えられた文が文法的であるかを回答してください。出力は以下から選択してください：<NL>" + "<NL>".join(list(JCoLA.CHOICES.values())) + "<NL>システム: 分かりました。<NL>"
    SEP = "<NL>"
    FEWSHOT_SEP = "<NL>"

    def doc_to_text(self, doc):
        input_text = doc["sentence"]
        return f"ユーザー: {input_text}{self.SEP}システム: "


class JCoLAWithRinnaBilingualInstructionSFT(JCoLAWithRinnaInstructionSFT):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/bilingual-gpt-neox-4b-instruction-sft
    """
    PROMPT_VERSION = 0.5
    DESCRIPTION = "ユーザー: " + f"与えられた文が文法的であるかを回答してください。出力は以下から選択してください：\n" + "\n".join(list(JCoLA.CHOICES.values())) + "\nシステム: 分かりました。\n"
    SEP = "\n"
    FEWSHOT_SEP = "\n"


VERSIONS = [
    JCoLA,
    JCoLAWithJAAlpacaPrompt,
    JCoLAWithRinnaInstructionSFT,
    JCoLAWithRinnaBilingualInstructionSFT
]

def construct_tasks():
    tasks = {}
    for version_class in VERSIONS:
        tasks[f"jcola-{version_class.VERSION}-{version_class.PROMPT_VERSION}"] = version_class
    return tasks
