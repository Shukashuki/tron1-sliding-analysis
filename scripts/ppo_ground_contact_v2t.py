"""T physical collector:matched-scale max-only control,unchanged sampling."""
from contact_duration_owner import namespace
_owner=namespace("T",__file__)
for _name in ("configure_sensors","clock","MARKER","SENSORS","GROUND","WHEELS","reward_term","install","GroundContactV2T"):
    globals()[_name]=_owner[_name]
