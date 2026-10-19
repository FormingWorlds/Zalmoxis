# EOS table registry check

Checks that a PALEOS table on disk is the release the fwl-io registry pins. `verify` hashes a table once per process (again only when its size or modification time changes) and compares the md5 with every md5 an installed fwl-io manifest pins for that file name. A table in its fwl-io version directory that differs is fetched again through fwl-io; a table anywhere else that differs raises `StaleTableError`. A file name that no PALEOS registry lists, such as a table of your own or a PALEOS-API table, is not checked. `StaleTableError` is not a `RuntimeError` or `ValueError`, so a caller that turns those into a failed structure solve does not hide it.

::: zalmoxis.eos.registry_guard
    options:
      inherited_members: false
      show_source: true
      members:
        - StaleTableError
        - verify
        - verify_digest
        - file_md5
