import 'package:flutter/material.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';

class FeedbackDialog extends StatelessWidget {
  const FeedbackDialog({super.key});

  static const _tags = [
    ('converted', Icons.star, Colors.amber),
    ('conversation', Icons.chat_bubble, Colors.blue),
    ('ignored', Icons.visibility_off, Colors.grey),
    ('negative', Icons.thumb_down, Colors.red),
    ('deleted', Icons.delete, Colors.orange),
  ];

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final labels = {
      'converted': l.feedbackConverted,
      'conversation': l.feedbackConversation,
      'ignored': l.feedbackIgnored,
      'negative': l.feedbackNegative,
      'deleted': l.feedbackDeleted,
    };

    return AlertDialog(
      title: Text(l.feedbackTitle),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: _tags.map((t) {
          final (tag, icon, color) = t;
          return ListTile(
            leading: Icon(icon, color: color, size: 22),
            title: Text(labels[tag] ?? tag),
            dense: true,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
            ),
            onTap: () => Navigator.of(context).pop(tag),
          );
        }).toList(),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(null),
          child: Text(l.cancel),
        ),
      ],
    );
  }
}
