# This handles collecting registration of all native ops
from . import ops
from .registry import register_all_operators


# Perform the outstanding registrations
register_all_operators()
