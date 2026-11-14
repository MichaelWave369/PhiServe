# Φ Inference Fabric R20: Portable Offline Field-Evidence Bundles

**R20 connects two real Windows field campaigns without requiring cloud upload.**
Its transport format is a plain, bounded, uncompressed ZIP containing only the
exact JSON source files produced by **PhiServe R18**. It is not a benchmark
engine, encrypted storage, a digital signature or independent device witness.

## Main-PC or backup-PC workflow

On each physical machine, first run a **real** operator-controlled R18
campaign. Follow [R18 instructions](OPERATOR_FIELD_CAMPAIGN_R18.md) and keep
the source JSON directory private. When the campaign has a complete
campaign-manifest.json, package it locally:

~~~powershell
phiserve-campaign-bundle pack --campaign "D:\PhiPrivate\backup-run-01" --output "D:\PhiPrivate\backup-run-01.zip"
phiserve-campaign-bundle check --archive "D:\PhiPrivate\backup-run-01.zip"
~~~

Use an operator-approved local transfer (for instance, removable media or
a trusted private share) to copy the ZIP onto the main PC. **R20 performs no
transfer, upload, remote connection or discovery**.

On the main computer, import to a **new private directory**, outside any
Git checkout. The destination must not already exist:

~~~powershell
phiserve-campaign-bundle check --archive "D:\PhiPrivate\backup-run-01.zip"
phiserve-campaign-bundle unpack --archive "D:\PhiPrivate\backup-run-01.zip" --out-dir "D:\PhiPrivate\imported-backup-run-01"
phiserve-campaign-review --campaign "D:\PhiPrivate\main-run-01" --campaign "D:\PhiPrivate\imported-backup-run-01" --output "D:\PhiPrivate\comparison-01.json"
~~~

The **check** action briefly uses a private local temporary directory to
revalidate the archive. It leaves no persistent imported campaign. **Unpack**
uses a private local staging directory, rechecks the extracted source files
and promotes the stage only after passing R19. On any exception it cleans up
its new staging directory and leaves the requested destination absent.

Never upload the archive or JSON to a public repo by default. These files
may reveal a model alias, hash, driver/runtime metadata and the operator's
hardware environment. The private ZIP is intentionally **not encrypted**:
choose the transport and storage permissions accordingly.

## Packing constraints

- R20 calls the merged R19 verifier on the selected source folder before
  creating any ZIP. R19 checks the R18 completion marker, original R13 trial
  receipts, SHA-256 file consistency, hardware snapshot and R15 advice.
- The only archive members are campaign-manifest.json, capacity-advice.json,
  hardware-snapshot.json, and the one-to-four explicitly listed
  benchmark-layers-N.json files. No extra files (including private
  notes, logs or credentials) are collected from the folder.
- ZIP entries are all top-level names. Source absolute paths, timestamps and
  arbitrary directory contents are not intentionally included.
- Output ZIP creation is exclusive and bounded to 8 MiB, with uncompressed
  member bytes. Invalid source or failed post-pack inspection removes only
  the newly created output.
- SHA-256 records and ZIP CRC guard accidental byte changes and
  inconsistencies. **Anyone who can modify source bytes can also
  recalculate unsigned hashes.** This system does not authenticate a
  device or author.

## Unpacking safeguards

Untrusted ZIPs are treated as data, never executed. R20 refuses:

- duplicate members, unexpected files, nested paths, drive-letter names,
  absolute paths or traversal attempts
- symlinked source archives or symlink/special file entries
- encrypted or compressed members, including deflate ZIP bombs
- source files over 1 MiB each or the manifest over 64 KiB
- archives over 8 MiB or more than seven members
- malformed, contradictory or falsely authoritative underlying R18, R13,
  PhiStream or R15 evidence
- destinations inside a Git checkout, destinations that already exist,
  or a symlinked destination parent

The archive is **not** a live deployment package, model distribution, proof
of real GPU execution, benchmark leaderboard, remote orchestrator or
attestation of separate machines. R19's same-model and same-experiment
checks still govern any two-campaign comparison, and a passing report never
approves inference routing or model promotion.

## Tests and true acceptance

~~~powershell
python -m unittest discover -s tests -p test_campaign_bundle.py -v
python -m unittest discover -s tests -v
~~~

The synthetic CI fixtures simulate R18 receipt files and malicious ZIPs.
They do not execute real GGUF weights and cannot measure the physical
RTX 5070 or RTX 3060. The real end-to-end acceptance step is to run R18 on
the machines, pack the backup evidence, transfer it privately and perform
an R19 review on the main PC.

**Evidence integrity above convenience. Operator consent above automation.**
