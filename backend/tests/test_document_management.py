import pytest
from fastapi import HTTPException

from app.api import upload as upload_api


def test_list_documents_returns_indexed_documents(monkeypatch):

    expected_documents = [
        {
            "filename": "sample.pdf",
            "chunks": 5
        }
    ]

    monkeypatch.setattr(
        upload_api,
        "list_documents",
        lambda: expected_documents
    )

    result = upload_api.get_documents()

    assert result["documents"] == expected_documents


def test_delete_document_successfully(tmp_path, monkeypatch):

    monkeypatch.setattr(
        upload_api,
        "UPLOAD_FOLDER",
        str(tmp_path)
    )

    document = tmp_path / "sample.pdf"
    document.write_bytes(b"%PDF-test")

    monkeypatch.setattr(
        upload_api,
        "delete_document",
        lambda filename: 5
    )

    result = upload_api.remove_document("sample.pdf")

    assert result["filename"] == "sample.pdf"
    assert result["chunks_deleted"] == 5
    assert not document.exists()


def test_delete_document_returns_404_when_missing(
    tmp_path,
    monkeypatch
):

    monkeypatch.setattr(
        upload_api,
        "UPLOAD_FOLDER",
        str(tmp_path)
    )

    with pytest.raises(HTTPException) as exc:
        upload_api.remove_document("missing.pdf")

    assert exc.value.status_code == 404


def test_delete_document_rejects_path_traversal():

    with pytest.raises(HTTPException) as exc:
        upload_api.remove_document("../outside.pdf")

    assert exc.value.status_code == 400


def test_delete_document_rejects_non_pdf_filename():

    with pytest.raises(HTTPException) as exc:
        upload_api.remove_document("sample.txt")

    assert exc.value.status_code == 400