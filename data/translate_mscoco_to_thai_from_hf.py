import argparse

from datasets import load_dataset
from fairseq.models.transformer import TransformerModel
from mosestokenizer import MosesTokenizer


ORG_DATASET_NAME = "HuggingFaceM4/COCO"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dataset_name", default="patomp/thai-mscoco-2014-captions")
    parser.add_argument("--subset", default="2014_captions")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model_name_or_path", default="SCB_1M+TBASE_en-th_moses-spm_130000-16000_v1.0")
    args = parser.parse_args()

    en_word_tokenize = MosesTokenizer('en')
    en2th_word2bpe = TransformerModel.from_pretrained(
        model_name_or_path='%s/models/'%args.model_name_or_path,
        checkpoint_file='checkpoint.pt',
        data_name_or_path='%s/vocab/'%args.model_name_or_path
    )
    def translate(en_captions: str) -> str:
        def _translate(_en_caption: str) -> str:
            __tokenized_sentence = ' '.join(en_word_tokenize(_en_caption))
            __th_caption = en2th_word2bpe.translate(__tokenized_sentence)
            return __th_caption.replace(' ', '').replace('▁', ' ').strip()
            
        outputs = []
        for en_caption in en_captions:
            if "\n" in en_caption:
                output = " ".join([_translate(s) for s in en_caption.split("\n")])
            else:
                output = _translate(en_caption)
            outputs.append(output)
        return outputs

    dataset = load_dataset(ORG_DATASET_NAME, args.subset, split=args.split)

    translated_dataset = dataset.map(
        lambda batch: {
            'th_sentences_raw': translate(batch["sentences_raw"])
        },
        batched=False
    )

    translated_dataset.push_to_hub(args.target_dataset_name, split=args.split)