import 'package:flutter/material.dart';

class BottomButton extends StatelessWidget {
  BottomButton({this.onTap,@required this.buttonTitle, this.borderRadius, this.colorBackground,this.colorTitle});
  final Function onTap;
  final String buttonTitle;
  final BorderRadius borderRadius;
  final Color colorBackground;
  final Color colorTitle;
  @override
  Widget build(context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Text(
              buttonTitle,
              style: TextStyle(
                color: colorTitle,
                fontFamily: 'OpenSans',
              ),
            )
          ],
        ),
        decoration: BoxDecoration(
          color: colorBackground,
          borderRadius: borderRadius,
        ),
      ),

    );
  }
}
