import 'package:flutter/material.dart';

final kHintTextStyle = TextStyle(
  color: Color(0xFFc5c8cf),
  fontFamily: 'OpenSans',
);

final kLabelStyle = TextStyle(
  color: Colors.white,
  fontWeight: FontWeight.bold,
  fontFamily: 'OpenSans',
);

final kBoxDecorationStyle = BoxDecoration(
  color: Colors.white,
  borderRadius: BorderRadius.circular(10.0),
  boxShadow: [
    BoxShadow(
      color: Colors.black12,
      blurRadius: 6.0,
      offset: Offset(0, 2),
    ),
  ],
);

final kLeftButtonBottomRadius = BorderRadius.only(
    topLeft: Radius.circular(10)
);
final kRightButtonBottomRadius = BorderRadius.only(
    topRight: Radius.circular(10)
);
//Color(0xFF6CA8F1)