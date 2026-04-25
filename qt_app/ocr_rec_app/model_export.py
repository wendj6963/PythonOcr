from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import hashes, hmac, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


@dataclass(frozen=True)
class ExportResult:
    onnx_path: Path
    enc_path: Path
    bytes_in: int
    bytes_out: int


@dataclass(frozen=True)
class ValidateResult:
    ok: bool
    message: str
    bytes_dec: int = 0
    sha256: str = ""


def _derive_key(passphrase: str, salt: bytes, length: int = 64) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        iterations=200_000,
    )
    return kdf.derive(passphrase.encode("utf-8"))


def _encrypt_aes_cbc_hmac(data: bytes, passphrase: str) -> bytes:
    salt = os.urandom(16)
    key = _derive_key(passphrase, salt, 64)
    enc_key = key[:32]
    mac_key = key[32:]
    iv = os.urandom(16)

    padder = padding.PKCS7(128).padder()
    padded = padder.update(data) + padder.finalize()

    cipher = Cipher(algorithms.AES(enc_key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded) + encryptor.finalize()

    mac = hmac.HMAC(mac_key, hashes.SHA256())
    mac.update(iv + ciphertext)
    tag = mac.finalize()

    # Format: MAGIC(4) + SALT(16) + IV(16) + TAG(32) + CIPHERTEXT
    return b"OCBC" + salt + iv + tag + ciphertext


def _decrypt_aes_cbc_hmac(data: bytes, passphrase: str) -> bytes:
    if len(data) < 4 + 16 + 16 + 32:
        raise ValueError("加密数据长度不足")
    if data[:4] != b"OCBC":
        raise ValueError("加密文件头不匹配")

    salt = data[4:20]
    iv = data[20:36]
    tag = data[36:68]
    ciphertext = data[68:]

    key = _derive_key(passphrase, salt, 64)
    enc_key = key[:32]
    mac_key = key[32:]

    mac = hmac.HMAC(mac_key, hashes.SHA256())
    mac.update(iv + ciphertext)
    mac.verify(tag)

    cipher = Cipher(algorithms.AES(enc_key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()

    unpadder = padding.PKCS7(128).unpadder()
    return unpadder.update(padded) + unpadder.finalize()


def export_to_onnx(
    model_path: Path,
    imgsz: int,
    out_dir: Path,
    out_name: str | None = None,
    nms: bool | None = None,
) -> Path:
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    export_kwargs: dict[str, object] = {
        "format": "onnx",
        "imgsz": imgsz,
        "opset": 12,
        "dynamic": False,
        "simplify": True,
    }
    if nms is not None:
        export_kwargs["nms"] = bool(nms)
    export = model.export(**export_kwargs)
    # ultralytics export returns a Path-like or string
    onnx_path = Path(str(export))
    if onnx_path.exists():
        if out_name:
            target = out_dir / out_name
            if onnx_path.resolve() != target.resolve():
                target.write_bytes(onnx_path.read_bytes())
            return target
        return onnx_path
    # fallback: guess in model dir
    guess = model_path.with_suffix(".onnx")
    if out_name:
        target = out_dir / out_name
        if guess.exists() and guess.resolve() != target.resolve():
            target.write_bytes(guess.read_bytes())
        return target
    return guess


def export_and_encrypt(
    model_path: Path,
    imgsz: int,
    out_dir: Path,
    passphrase: str,
    onnx_name: str | None = None,
    enc_name: str | None = None,
    nms: bool | None = None,
) -> ExportResult:
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    if model_path.suffix.lower() == ".onnx":
        onnx_path = model_path
        if onnx_name:
            target = out_dir / onnx_name
            if onnx_path.resolve() != target.resolve():
                target.write_bytes(onnx_path.read_bytes())
            onnx_path = target
    else:
        onnx_path = export_to_onnx(model_path, imgsz, out_dir, out_name=onnx_name, nms=nms)
        if not onnx_path.exists():
            raise FileNotFoundError("导出 ONNX 失败，请检查模型与依赖")

    data = onnx_path.read_bytes()
    encrypted = _encrypt_aes_cbc_hmac(data, passphrase)

    if not enc_name:
        enc_name = onnx_path.stem + ".onnx.enc"
    enc_path = out_dir / enc_name
    enc_path.write_bytes(encrypted)

    return ExportResult(
        onnx_path=onnx_path,
        enc_path=enc_path,
        bytes_in=len(data),
        bytes_out=len(encrypted),
    )


def validate_encrypted_model(
    enc_path: Path,
    passphrase: str,
    onnx_path: Optional[Path] = None,
) -> ValidateResult:
    if not enc_path.exists():
        return ValidateResult(False, f"加密文件不存在: {enc_path}")
    try:
        enc_data = enc_path.read_bytes()
        dec_data = _decrypt_aes_cbc_hmac(enc_data, passphrase)
    except Exception as exc:
        return ValidateResult(False, f"解密失败: {exc}")

    sha256 = hashes.Hash(hashes.SHA256())
    sha256.update(dec_data)
    digest = sha256.finalize().hex()

    if onnx_path and onnx_path.exists():
        try:
            onnx_data = onnx_path.read_bytes()
        except Exception as exc:
            return ValidateResult(False, f"读取 ONNX 失败: {exc}")
        if onnx_data != dec_data:
            return ValidateResult(False, "解密内容与 ONNX 文件不一致")

    if len(dec_data) == 0:
        return ValidateResult(False, "解密结果为空")

    return ValidateResult(True, "验证通过", bytes_dec=len(dec_data), sha256=digest)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="导出 ONNX 并 AES-256-CBC+HMAC 加密")
    parser.add_argument("--model", required=True, help="模型路径（.pt 或 .onnx）")
    parser.add_argument("--imgsz", type=int, default=640, help="导出 imgsz")
    parser.add_argument("--out", default="qt_app/ocr_rec_app/models", help="输出目录")
    parser.add_argument("--key", required=True, help="加密 KEY")
    parser.add_argument("--nms", action="store_true", help="导出时启用 NMS")
    args = parser.parse_args()

    res = export_and_encrypt(
        Path(args.model),
        int(args.imgsz),
        Path(args.out),
        args.key,
        nms=bool(args.nms),
    )
    print(f"ONNX: {res.onnx_path}")
    print(f"ENC : {res.enc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
