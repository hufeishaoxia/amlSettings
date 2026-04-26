"""
Test script for Azure OpenAI gpt-5.1 deployment on `msncompanionce`.

Resource (parsed from portal URL):
  Subscription : b6dc87f3-c479-49c8-8cb5-7896da3ff895
  ResourceGroup: DevTestRG
  Account      : msncompanionce
  Deployment   : gpt-5.1
  Tenant       : 72f988bf-86f1-41af-91ab-2d7cd011db47 (Microsoft)

Usage:
  pip install "openai>=1.55.0" "azure-identity>=1.19.0"
  az login --tenant 72f988bf-86f1-41af-91ab-2d7cd011db47
  python test_gpt51.py                     # default prompt
  python test_gpt51.py "your question"     # custom prompt

Auth: tries AAD (DefaultAzureCredential) first, falls back to AZURE_OPENAI_API_KEY env var.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

ENDPOINT = "https://msncompanionce.openai.azure.com/"
DEPLOYMENT = "gpt-5.1"
API_VERSION = "2024-10-21"   # stable GA; bump to a preview if you need new features
TENANT_ID = "72f988bf-86f1-41af-91ab-2d7cd011db47"
SCOPE = "https://cognitiveservices.azure.com/.default"


def build_client():
    """Return an AzureOpenAI client, preferring AAD, falling back to API key."""
    from openai import AzureOpenAI

    # --- Try AAD first ---
    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        cred = DefaultAzureCredential(
            interactive_browser_tenant_id=TENANT_ID,
            shared_cache_tenant_id=TENANT_ID,
            visual_studio_code_tenant_id=TENANT_ID,
            exclude_interactive_browser_credential=False,
        )
        # Probe the token early so we fail fast with a clear error.
        _ = cred.get_token(SCOPE)
        token_provider = get_bearer_token_provider(cred, SCOPE)
        print(f"[auth] using AAD (DefaultAzureCredential, tenant={TENANT_ID})")
        return AzureOpenAI(
            azure_endpoint=ENDPOINT,
            azure_ad_token_provider=token_provider,
            api_version=API_VERSION,
        )
    except Exception as e:
        print(f"[auth] AAD failed: {type(e).__name__}: {e}")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not api_key:
            print(
                "[auth] no AZURE_OPENAI_API_KEY env var either. "
                "Either run `az login --tenant 72f988bf-86f1-41af-91ab-2d7cd011db47` "
                "or set AZURE_OPENAI_API_KEY."
            )
            raise
        print("[auth] falling back to API key")
        return AzureOpenAI(
            azure_endpoint=ENDPOINT,
            api_key=api_key,
            api_version=API_VERSION,
        )


def run_sync(client, prompt: str) -> None:
    print("\n" + "=" * 60)
    print("SYNC chat.completions")
    print("=" * 60)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=DEPLOYMENT,  # for AzureOpenAI, `model` is the deployment name
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=512,
    )
    dt = time.time() - t0
    msg = resp.choices[0].message
    print(f"[ok] latency: {dt:.2f}s")
    print(f"[ok] finish_reason: {resp.choices[0].finish_reason}")
    if resp.usage:
        print(
            f"[ok] tokens: prompt={resp.usage.prompt_tokens} "
            f"completion={resp.usage.completion_tokens} "
            f"total={resp.usage.total_tokens}"
        )
    print("\n--- response ---")
    print(msg.content)


def run_stream(client, prompt: str) -> None:
    print("\n" + "=" * 60)
    print("STREAM chat.completions")
    print("=" * 60)
    t0 = time.time()
    first_token_at = None
    chunks = 0
    print("--- streaming response ---")
    stream = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=512,
        stream=True,
    )
    for ev in stream:
        if not ev.choices:
            continue
        delta = ev.choices[0].delta
        if delta and delta.content:
            if first_token_at is None:
                first_token_at = time.time() - t0
            chunks += 1
            sys.stdout.write(delta.content)
            sys.stdout.flush()
    total = time.time() - t0
    print(
        f"\n[ok] ttft={first_token_at:.2f}s  total={total:.2f}s  chunks={chunks}"
        if first_token_at is not None
        else f"\n[warn] no content streamed. total={total:.2f}s"
    )


def main() -> int:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "用一句话解释相对论。"
    print(f"[cfg] endpoint   : {ENDPOINT}")
    print(f"[cfg] deployment : {DEPLOYMENT}")
    print(f"[cfg] api_version: {API_VERSION}")
    print(f"[cfg] prompt     : {prompt!r}")

    try:
        client = build_client()
    except Exception:
        traceback.print_exc()
        print("\n[hint] AAD not working? Try:")
        print(f"  az login --tenant {TENANT_ID}")
        print("  az account set --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895")
        print("Or use API key:")
        print("  $env:AZURE_OPENAI_API_KEY = '<key from portal -> Keys and Endpoint>'")
        return 2

    try:
        run_sync(client, prompt)
        run_stream(client, prompt)
    except Exception as e:
        print(f"\n[err] request failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\n[troubleshoot]")
        print("  401 / PermissionDenied -> your AAD identity needs role")
        print("                            'Cognitive Services OpenAI User' on the resource")
        print("                            (Portal -> msncompanionce -> Access control (IAM))")
        print("  404 DeploymentNotFound -> check deployment name is exactly 'gpt-5.1'")
        print("  429                   -> quota/TPM exhausted, retry or raise quota")
        print("  unsupported parameter -> bump API_VERSION to a newer preview")
        return 1

    print("\n[done] all good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())