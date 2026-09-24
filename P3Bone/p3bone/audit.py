"""Patient provenance checks for user-supplied manifests."""
from .calibration import validate_folds
from .utils import require


def audit_splits(cohorts):
    groups = {name: {r['patient_id'] for r in rows} for name, rows in cohorts.items()}
    require({'calibration', 'unlabeled', 'test'} <= set(groups), 'Missing primary cohorts')
    validate_folds(cohorts['calibration'])
    overlaps = {}
    for i, first in enumerate(groups):
        for second in list(groups)[i+1:]:
            common = groups[first] & groups[second]
            overlaps[first+'__'+second] = sorted(common)
            if 'test' in (first, second) or {first, second} == {'calibration', 'unlabeled'}:
                require(not common, f'Patient overlap between {first} and {second}')
    return {'status': 'PASSED', 'counts': {k: {'images': len(v), 'patients': len(groups[k])} for k, v in cohorts.items()},
            'overlaps': overlaps, 'note': 'Historical labeled-cohort overlap is reported, not concealed. '
            'sample-prompts additionally verifies that routed localizer models did not train on the current patient. '
            'This audit cannot establish unseen pretraining provenance or recover unpublished original patient lists.'}
