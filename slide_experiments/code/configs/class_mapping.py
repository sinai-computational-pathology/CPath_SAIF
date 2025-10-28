# Class label mappings configuration
# Place project-specific standard mappings here so training/validation can reuse.

# Map demographic keys to canonical class label lists
DEMOGRAPHIC_LABELS = {
    'self_reported': ['White', 'Black', 'Hispanic/Latino', 'Asian', 'Other'],
    'genetic_ancestry': ['White', 'Black', 'Hispanic/Latino', 'Asian', 'Other'],
    'sex': ['Male', 'Female'],
    # Add other demographic mappings here as needed, e.g.:
    # 'age_group': ['young', 'middle', 'old']
}


def get_class_labels(demo_key=None, num_classes=None):
    """Return list of class label names.

    Priority:
    - If demo_key is provided and a mapping exists, return that mapping.
    - Else if num_classes provided and a numeric mapping exists, return that mapping.
    - Else return generic names class_0..class_{n-1} (default n=5).
    """
    # Prefer explicit demographic key
    if demo_key is not None:
        key = str(demo_key)
        if key in DEMOGRAPHIC_LABELS:
            return DEMOGRAPHIC_LABELS[key]
    # Fallback to numeric mapping
    n = int(num_classes)
    # If numeric maps to a known demographic mapping, return it
    for k, v in DEMOGRAPHIC_LABELS.items():
        if len(v) == n:
            return v
    # Generic fallback
    return [f'class_{i}' for i in range(n)]
