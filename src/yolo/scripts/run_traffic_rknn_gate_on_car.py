#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Upload val/test temporarily, run RKNNLite gates on the car, download reports."""

from __future__ import annotations

import argparse
import os
import posixpath
import shlex
import tarfile
import tempfile
from pathlib import Path


def execute(client, command, allow_failure=False):
    stdin, stdout, stderr = client.exec_command(command)
    del stdin
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    if err:
        print(err, end="" if err.endswith("\n") else "\n")
    if code != 0 and not allow_failure:
        raise RuntimeError("remote command failed ({}): {}".format(code, command))
    return code


def upload(sftp, local_path, remote_path):
    print("[UPLOAD] {} -> {}".format(local_path, remote_path))
    sftp.put(str(local_path), remote_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.1.6")
    parser.add_argument("--user", default="ucar")
    parser.add_argument("--password-env", default="UCAR_SSH_PASSWORD")
    parser.add_argument("--data", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--compare-script", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keep-remote", action="store_true")
    args = parser.parse_args()

    password = os.environ.get(args.password_env)
    if not password:
        raise RuntimeError("missing password environment variable {}".format(args.password_env))
    data_root = Path(args.data).expanduser().resolve()
    reference = Path(args.reference).expanduser().resolve()
    compare_script = Path(args.compare_script).expanduser().resolve()
    models = [Path(item).expanduser().resolve() for item in args.model]
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for path in [reference, compare_script] + models:
        if not path.is_file():
            raise FileNotFoundError(str(path))

    remote_root = "/tmp/traffic_rknn_gate_codex"
    if not remote_root.startswith("/tmp/traffic_rknn_gate_"):
        raise RuntimeError("unsafe remote temporary path")

    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, username=args.user, password=password, timeout=10)
    archive_name = None
    try:
        execute(
            client,
            "python3 -c \"import sys, rknnlite; print(sys.version); print(rknnlite.__file__)\"; "
            "df -h /tmp /home/ucar; test -d /home/ucar/2026-xunfei-race && echo WORKSPACE_OK",
        )
        execute(client, "rm -rf {0} && mkdir -p {0}/data {0}/output".format(shlex.quote(remote_root)))
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as handle:
            archive_name = handle.name
        with tarfile.open(archive_name, "w:gz") as archive:
            for split in ("val", "test"):
                archive.add(str(data_root / split), arcname=split)

        sftp = client.open_sftp()
        try:
            remote_archive = posixpath.join(remote_root, "data.tar.gz")
            upload(sftp, Path(archive_name), remote_archive)
            upload(sftp, reference, posixpath.join(remote_root, "reference.csv"))
            upload(sftp, compare_script, posixpath.join(remote_root, "compare.py"))
            for model in models:
                upload(sftp, model, posixpath.join(remote_root, model.name))
        finally:
            sftp.close()
        execute(
            client,
            "tar -xzf {0}/data.tar.gz -C {0}/data".format(shlex.quote(remote_root)),
        )

        for model in models:
            variant = "int8" if "int8" in model.stem.lower() else "fp16"
            command = (
                "python3 {root}/compare.py --data {root}/data --reference {root}/reference.csv "
                "--rknn {root}/{model} --output {root}/output/{variant}"
            ).format(
                root=shlex.quote(remote_root),
                model=shlex.quote(model.name),
                variant=shlex.quote(variant),
            )
            execute(client, command, allow_failure=True)

        sftp = client.open_sftp()
        try:
            for model in models:
                variant = "int8" if "int8" in model.stem.lower() else "fp16"
                local_dir = output_root / variant
                local_dir.mkdir(parents=True, exist_ok=True)
                for filename in ("report.json", "per_image.csv"):
                    remote_path = posixpath.join(remote_root, "output", variant, filename)
                    local_path = local_dir / filename
                    sftp.get(remote_path, str(local_path))
                    print("[DOWNLOAD] {}".format(local_path))
        finally:
            sftp.close()
    finally:
        if archive_name:
            try:
                os.unlink(archive_name)
            except OSError:
                pass
        if not args.keep_remote:
            execute(client, "rm -rf {}".format(shlex.quote(remote_root)), allow_failure=True)
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
