/// A piece of office furniture with relative (0-1) coordinates so it scales
/// to any canvas size.
class Furniture {
  const Furniture({
    required this.type,
    required this.x,
    required this.y,
    required this.w,
    required this.h,
  });

  final String type; // desk, server, board, plant, coffee
  final double x;
  final double y;
  final double w;
  final double h;
}

/// Default office layout using relative coordinates.
const List<Furniture> officeFurniture = [
  Furniture(type: 'desk', x: 0.15, y: 0.6, w: 0.12, h: 0.1),
  Furniture(type: 'desk', x: 0.42, y: 0.45, w: 0.12, h: 0.1),
  Furniture(type: 'desk', x: 0.68, y: 0.65, w: 0.12, h: 0.1),
  Furniture(type: 'server', x: 0.87, y: 0.25, w: 0.06, h: 0.18),
  Furniture(type: 'board', x: 0.04, y: 0.2, w: 0.1, h: 0.14),
  Furniture(type: 'plant', x: 0.92, y: 0.78, w: 0.04, h: 0.12),
  Furniture(type: 'plant', x: 0.02, y: 0.82, w: 0.04, h: 0.12),
  Furniture(type: 'coffee', x: 0.56, y: 0.2, w: 0.05, h: 0.08),
];
