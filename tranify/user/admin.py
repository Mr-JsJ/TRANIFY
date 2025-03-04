from django.contrib import admin
from .models import TrainedModel

@admin.register(TrainedModel)
class TrainedModelAdmin(admin.ModelAdmin):
    list_display = ('project_name', 'user', 'num_classes', 'epochs', 'created_at')
    search_fields = ('project_name', 'user__username')
    list_filter = ('created_at', 'epochs')
    readonly_fields = ('created_at',)

    def get_readonly_fields(self, request, obj=None):
        """Make certain fields read-only for existing instances."""
        if obj:  # Editing an existing object
            return self.readonly_fields + ('user',)
        return self.readonly_fields
