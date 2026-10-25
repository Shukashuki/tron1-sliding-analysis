"""T:twice worst8 continuous any-wheel deficit,matched scale control for S."""
import ground_contact_duration_cost as common
base=common.base
BASE_F_DEFINITION_SHA256=common.BASE_F_DEFINITION_SHA256
BASE_G_DEFINITION_SHA256=common.BASE_G_DEFINITION_SHA256
BASE_I_DEFINITION_SHA256=common.BASE_I_DEFINITION_SHA256
DEFAULT_SETTINGS=common.settings("T")
verify_base=common.verify_base
def validate_settings(settings=None):return common.settings("T",settings)
def formula_metadata(settings=None):return common.metadata("T",settings)
def reward_numpy(*args,settings=None):return common.reward_numpy("T",*args,settings_value=settings)
def reward_torch(*args,settings=None,validate=True):return common.reward_torch("T",*args,settings_value=settings,validate=validate)
