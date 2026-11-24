from io import BytesIO
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.uploader import upload_service


def create_upload_file(filename, content=b"%PDF-test-content"):
    return UploadFile(
        filename=filename,
        file=BytesIO(content)
    )


def test_upload_pdf_successfully(tmp_path, monkeypatch):

    monkeypatch.setattr(
        upload_service,
        "UPLOAD_FOLDER",
        tmp_path
    )

    monkeypatch.setattr(
        upload_service,
        "ingest_document",
        lambda path: 3
    )

    file = create_upload_file("sample.pdf")

    result = upload_service.upload_pdf(file)

    assert result["filename"] == "sample.pdf"
    assert result["chunks"] == 3
    assert (tmp_path / "sample.pdf").exists()


def test_upload_rejects_missing_filename():

    file = create_upload_file("")

    with pytest.raises(HTTPException) as exc:
        upload_service.upload_pdf(file)

    assert exc.value.status_code == 400


def test_upload_rejects_non_pdf_file():

    file = create_upload_file("sample.txt")

    with pytest.raises(HTTPException) as exc:
        upload_service.upload_pdf(file)

    assert exc.value.status_code == 400


def test_upload_rejects_duplicate_filename(tmp_path, monkeypatch):

    monkeypatch.setattr(
        upload_service,
        "UPLOAD_FOLDER",
        tmp_path
    )

    (tmp_path / "sample.pdf").write_bytes(b"existing")

    file = create_upload_file("sample.pdf")

    with pytest.raises(HTTPException) as exc:
        upload_service.upload_pdf(file)

    assert exc.value.status_code == 409


def test_upload_cleans_up_after_ingestion_failure(
    tmp_path,
    monkeypatch
):

    monkeypatch.setattr(
        upload_service,
        "UPLOAD_FOLDER",
        tmp_path
    )

    mock_delete = Mock()

    monkeypatch.setattr(
        upload_service,
        "delete_document",
        mock_delete
    )

    def fail_ingestion(path):
        raise RuntimeError("Ingestion failed")

    monkeypatch.setattr(
        upload_service,
        "ingest_document",
        fail_ingestion
    )

    file = create_upload_file("failed.pdf")

    with pytest.raises(RuntimeError):
        upload_service.upload_pdf(file)

    assert not (tmp_path / "failed.pdf").exists()

    mock_delete.assert_called_once_with("failed.pdf")


def test_upload_normalizes_path_in_filename(tmp_path, monkeypatch):

    monkeypatch.setattr(
        upload_service,
        "UPLOAD_FOLDER",
        tmp_path
    )

    monkeypatch.setattr(
        upload_service,
        "ingest_document",
        lambda path: 2
    )

    file = create_upload_file("../../sample.pdf")

    result = upload_service.upload_pdf(file)

    assert result["filename"] == "sample.pdf"
    assert (tmp_path / "sample.pdf").exists()