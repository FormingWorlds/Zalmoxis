# EOS table registry check

Checks that a PALEOS table on disk is the release the fwl-io registry pins. `verify` hashes a table once per process (again when the file changes on disk) and compares the digest with the pins an installed fwl-io manifest gives its file name and the name of the file it resolves to: of the two names, only those a registry pins are checked, and the digest must match a pin common to all of them, compared case-insensitively. Every pin must be an md5; a pin in another form stops the read and names the pin. Any failure of the check itself is a `StaleTableError` too. A table in its version directory under the current `FWL_DATA` that differs is fetched again through fwl-io; a table anywhere else that differs raises `StaleTableError`. A file name that no PALEOS registry lists, such as a table of your own or a PALEOS-API table, is not checked. `StaleTableError` is not a `RuntimeError` or `ValueError`, so a caller that turns those into a failed structure solve does not hide it.

::: zalmoxis.eos.registry_guard
    options:
      inherited_members: false
      show_source: true
      members:
        - StaleTableError
        - verify
        - verify_digest
        - file_md5
