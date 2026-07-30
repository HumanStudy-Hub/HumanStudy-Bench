import unittest

from generation_pipeline.pdf.models import DocumentBlock, ParsedPdfDocument
from generation_pipeline.pdf.parser import (
    _repair_ambiguous_decimal_tokens,
    _repair_ambiguous_statistic_tokens,
    _repair_systematic_text_layer_artifacts,
)


def _document(texts):
    return ParsedPdfDocument(
        source_file="paper.pdf",
        source_sha256="hash",
        parser="test",
        parser_version="1",
        page_count=len(texts),
        blocks=[
            DocumentBlock(
                block_id=f"b{index}",
                order=index,
                page_start=index + 1,
                page_end=index + 1,
                block_type="text",
                text=text,
            )
            for index, text in enumerate(texts)
        ],
        markdown="\n".join(texts),
    )


class PdfParserRepairTests(unittest.TestCase):
    def test_repairs_ambiguous_decimal_only_with_aligned_second_ocr_context(self):
        source = (
            "A realistic estimate was below .50. The median estimate of the "
            "respondents was as high as .S5. This confidence affected planning."
        )
        ocr = (
            "A realistic estimate was below .50. The median estimate of the "
            "respondents was as high as .85. This confidence affected planning."
        )

        repaired, repairs = _repair_ambiguous_decimal_tokens(source, ocr)

        self.assertIn("as high as .85", repaired)
        self.assertEqual(repairs[0]["original"], ".S5")
        self.assertEqual(repairs[0]["replacement"], ".85")

    def test_keeps_ambiguous_decimal_when_second_ocr_context_does_not_match(self):
        source = "The median estimate of respondents was .S5 in this task."
        unrelated_ocr = "A separate table reports a probability of .85 for another outcome."

        repaired, repairs = _repair_ambiguous_decimal_tokens(source, unrelated_ocr)

        self.assertEqual(repaired, source)
        self.assertEqual(repairs, [])

    def test_repairs_decimal_when_second_ocr_drops_only_the_decimal_point(self):
        source = "The sequence was significantly less likely (p < .Ol) in the same sample."
        ocr = "The sequence was significantly less likely (p < 01) in the same sample."

        repaired, repairs = _repair_ambiguous_decimal_tokens(source, ocr)

        self.assertIn("p < .01", repaired)
        self.assertEqual(repairs[0]["replacement"], ".01")

    def test_repairs_dropped_statistic_decimal_with_aligned_second_ocr(self):
        source = (
            "The experiment confirmed the theory (x = 223, p < .05, two-tailed) "
            "before the replication question."
        )
        ocr = (
            "The experiment confirmed the theory (z = 2.23, p < .05, two-tailed) "
            "before the replication question."
        )

        repaired, repairs = _repair_ambiguous_statistic_tokens(source, ocr)

        self.assertIn("z = 2.23", repaired)
        self.assertEqual(repairs[0]["original"], "x = 223")

    def test_repairs_repeated_bracket_glyph_and_impossible_probability_fraction(self):
        document = _document(
            [
                "Problem 1 [N = 1521: option with 113 probability 128 percent]",
                "Problem 2 [N=1551: option with 213 probabilities 113 percent]",
                "Problem 3 [ N = 861. another task",
            ]
        )
        repaired = _repair_systematic_text_layer_artifacts(document)
        self.assertIn("[N = 152]:", repaired.blocks[0].text)
        self.assertIn("1/3 probability", repaired.blocks[0].text)
        self.assertIn("2/3 probabilities", repaired.blocks[1].text)
        self.assertIn("[ N = 86].", repaired.blocks[2].text)
        self.assertIn("[28 percent]", repaired.blocks[0].text)
        self.assertIn("[13 percent]", repaired.blocks[1].text)
        self.assertIn("[28 percent]", repaired.markdown)
        self.assertIn("systematic_n_bracket_glyph_repaired:3", repaired.warnings)
        self.assertIn("systematic_fraction_glyph_repaired:2", repaired.warnings)
        self.assertIn("systematic_bracket_percent_glyph_repaired:2", repaired.warnings)

    def test_does_not_rewrite_a_single_sample_size_ending_in_one(self):
        document = _document(["Study 1 [N = 1521: one isolated occurrence"])
        repaired = _repair_systematic_text_layer_artifacts(document)
        self.assertIn("[N = 1521:", repaired.blocks[0].text)
        self.assertFalse(repaired.warnings)

    def test_does_not_rewrite_a_well_formed_sample_size_ending_in_one(self):
        document = _document(
            [
                "Problem 1 [N = 1521: broken",
                "Problem 2 [N = 1551: broken",
                "Problem 3 [N = 861: broken",
                "A real study reports [N = 81].",
            ]
        )
        repaired = _repair_systematic_text_layer_artifacts(document)
        self.assertIn("[N = 81].", repaired.blocks[3].text)

    def test_repairs_bracketed_currency_i_and_missing_closing_bracket(self):
        document = _document(
            ["The item cost ($10) [$I201 at the other branch of the store."]
        )
        repaired = _repair_systematic_text_layer_artifacts(document)
        self.assertIn("($10) [$120] at", repaired.blocks[0].text)
        self.assertIn("($10) [$120] at", repaired.markdown)
        self.assertIn("systematic_currency_bracket_glyph_repaired:1", repaired.warnings)


if __name__ == "__main__":
    unittest.main()
