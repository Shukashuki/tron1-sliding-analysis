"""S physical collector:all eight actual samples,unchanged lifecycle and beta."""
from contact_duration_owner import namespace
_owner=namespace("S",__file__)
for _name in ("configure_sensors","clock","MARKER","SENSORS","GROUND","WHEELS","reward_term","install","GroundContactV2S"):
    globals()[_name]=_owner[_name]
