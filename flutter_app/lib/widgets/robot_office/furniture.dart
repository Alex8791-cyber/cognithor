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

  final String type; // desk, server, board, plant, coffee, flower, hanging_plant
  final double x;
  final double y;
  final double w;
  final double h;
}

/// Default office layout using relative coordinates.
/// Open-plan office: 5 desks in staggered rows for realistic spacing.
const List<Furniture> officeFurniture = [
  // Row 1 (back) — 2 desks
  Furniture(type: 'desk', x: 0.18, y: 0.52, w: 0.12, h: 0.1),
  Furniture(type: 'desk', x: 0.42, y: 0.50, w: 0.12, h: 0.1),
  // Row 2 (front) — 3 desks
  Furniture(type: 'desk', x: 0.12, y: 0.66, w: 0.12, h: 0.1),
  Furniture(type: 'desk', x: 0.38, y: 0.64, w: 0.12, h: 0.1),
  Furniture(type: 'desk', x: 0.64, y: 0.66, w: 0.12, h: 0.1),
  // Infrastructure
  Furniture(type: 'server', x: 0.87, y: 0.25, w: 0.06, h: 0.18),
  Furniture(type: 'board', x: 0.04, y: 0.2, w: 0.1, h: 0.14),
  // Plants
  Furniture(type: 'plant', x: 0.92, y: 0.78, w: 0.04, h: 0.12),
  Furniture(type: 'plant', x: 0.02, y: 0.82, w: 0.04, h: 0.12),
  Furniture(type: 'plant', x: 0.10, y: 0.80, w: 0.04, h: 0.12),
  Furniture(type: 'hanging_plant', x: 0.70, y: 0.12, w: 0.06, h: 0.10),
  // Coffee area
  Furniture(type: 'coffee', x: 0.56, y: 0.2, w: 0.05, h: 0.08),
  // Flowers
  Furniture(type: 'flower', x: 0.13, y: 0.48, w: 0.03, h: 0.06),
  Furniture(type: 'flower', x: 0.75, y: 0.49, w: 0.03, h: 0.06),
  Furniture(type: 'flower', x: 0.27, y: 0.32, w: 0.03, h: 0.06),
  Furniture(type: 'flower', x: 0.59, y: 0.31, w: 0.03, h: 0.06),
];
