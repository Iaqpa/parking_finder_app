import 'package:flutter/material.dart';

class BottomSheetShape extends ShapeBorder {
  @override
  // TODO: implement dimensions
  EdgeInsetsGeometry get dimensions => throw UnimplementedError();

  @override
  Path getInnerPath(Rect rect, {TextDirection textDirection}) {
    // TODO: implement getInnerPath
    throw UnimplementedError();
  }

  @override
  Path getOuterPath(Rect rect, {TextDirection textDirection}) {
    return getClip(rect.size);
  }

  @override
  void paint(Canvas canvas, Rect rect, {TextDirection textDirection}) {
    // TODO: implement paint
  }

  @override
  ShapeBorder scale(double t) {
    // TODO: implement scale
    throw UnimplementedError();
  }

  Path getClip(Size size) {
    Path path = Path();

    path.moveTo(0, 40);
    path.addRRect(RRect.fromRectAndCorners(Rect.fromLTRB(0, size.height, size.width, 0),topLeft: Radius.circular(25),topRight: Radius.circular(25)));
    //path.quadraticBezierTo(size.width/2, 0size.width/2, 0, size.width, 40);
    return path;
  }

}