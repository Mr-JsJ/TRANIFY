from django import template

register = template.Library()

@register.filter
def get(dictionary, key):
    """Fetches the value from a dictionary using a dynamic key."""
    return dictionary.get(key, "N/A")  # Returns 'N/A' if key doesn't exist

