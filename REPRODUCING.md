# Reproducing

Each release carries its own replay instructions; this file indexes
them. The general shape is constant across releases:

1. check out the release tag;
2. create the pinned Python environment (a lockfile ships with each
   release);
3. run the release's certificate programs; each prints its executed
   check count and writes a canonical JSON artifact;
4. compare the regenerated artifacts against the committed ones —
   they must match byte for byte;
5. optionally rebuild the PDFs from the released sources and compare
   hashes against the release manifest.

No release is public yet; the first release's exact commands will be
recorded here when it ships.
