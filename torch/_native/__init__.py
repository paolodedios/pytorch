from .registry import register_all_operators

# This handles collecting registration of all native ops
from . import ops

# Perform the outstanding registrations
register_all_operators()
