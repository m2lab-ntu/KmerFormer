"""Genus-only off-catalogue manifests must work without fake species labels."""
import pytest

from kmerformer.data_loader import load_test_data


def test_genus_only_labels_keep_original_class_ids(tmp_path):
    fasta = tmp_path / "reads.fa"
    labels = tmp_path / "labels.tsv"
    fasta.write_text(">a\nACGT\n>b\nTGCA\n")
    labels.write_text("seq_id\tgenus_class\tgenus_name\n"
                      "a\t7\tA\nb\t119\tB\n")
    data = load_test_data(str(fasta), str(labels), task="genus")
    assert data["val_genus_labels"].tolist() == [7, 119]
    assert data["num_genera"] == 2
    assert data["num_species"] == 0
    assert data["val_species_labels"].tolist() == [-1, -1]
    assert data["id2species"] == {}
    with pytest.raises(ValueError, match="species_class"):
        load_test_data(str(fasta), str(labels), task="species")
