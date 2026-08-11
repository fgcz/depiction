Only `parse_metadata.py` is left of the hand-rolled imzML parser that used to do the reading;
imzy replaced the rest in Phase E of the refactoring. It survives because imzy parses no
checksums and reports a pixel size of `1` where a file declares none, so metadata still comes
from here. The streaming Rust rewrite an earlier version of this file described was never
started.
