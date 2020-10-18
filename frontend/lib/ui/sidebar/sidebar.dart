import 'dart:async';

import 'package:flutter/material.dart';
import 'package:rxdart/rxdart.dart';


class SideBar extends StatefulWidget {
  SideBar({this.searchByMyPos, this.searchByAddress});
  final Function searchByMyPos;
  final Function searchByAddress;
  @override
  _SideBarState createState() => _SideBarState();
}

class _SideBarState extends State<SideBar>
    with SingleTickerProviderStateMixin<SideBar> {
  AnimationController _animationController;
  StreamController<bool> isSidebarOpenedStreamController;
  Stream<bool> isSidebarOpenedStream;
  StreamSink<bool> isSidebarOpenedSink;
  final _animationDuration = const Duration(milliseconds: 500);

  @override
  void initState() {
    super.initState();
    _animationController =
        AnimationController(vsync: this, duration: _animationDuration);
    isSidebarOpenedStreamController = PublishSubject<bool>();
    isSidebarOpenedStream = isSidebarOpenedStreamController.stream;
    isSidebarOpenedSink = isSidebarOpenedStreamController.sink;
  }

  @override
  void dispose() {
    _animationController.dispose();
    isSidebarOpenedStreamController.close();
    isSidebarOpenedSink.close();
    super.dispose();
  }

  void onIconPressed() {
    final animationStatus = _animationController.status;
    final isAnimationCompleted = animationStatus == AnimationStatus.completed;

    if (isAnimationCompleted) {
      isSidebarOpenedSink.add(false);
      _animationController.reverse();
    } else {
      isSidebarOpenedSink.add(true);
      _animationController.forward();
    }
  }

  @override
  Widget build(BuildContext context) {
    final screenHeight = MediaQuery.of(context).size.height;
    return StreamBuilder<bool>(
      initialData: false,
      stream: isSidebarOpenedStream,
      builder: (context, isSideBarOpenedAsync) {
        return AnimatedPositioned(
          duration: _animationDuration,
          top: isSideBarOpenedAsync.data ? 0 : -screenHeight,
          bottom:
              isSideBarOpenedAsync.data ? screenHeight / 2: screenHeight - 80,
          left: 0,
          right: 0,
          /*duration: _animationDuration,
          top: 0,
          bottom: 0,
          left: isSideBarOpenedAsync.data ? 0 : -screenWidth,
          right: isSideBarOpenedAsync.data ? 0 : screenWidth-100,*/
          child: Column(
            children: <Widget>[
              Expanded(
                child: Container(
                  padding: const EdgeInsets.only(
                    top: 45,
                    bottom: 16,
                    left: 32,
                    right: 32,
                  ),
                  color: Color(0xFF29304a),
                  child: Column(
                    mainAxisSize: MainAxisSize.max,
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: <Widget>[
                      Text(
                        "Где хотите искать свободные места?",
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      Divider(
                        height: 10,
                        thickness: 0.5,
                        color: Color(0xFF23eacb),
                        indent: 0,
                        endIndent: 0,
                      ),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,

                        children: <Widget>[
                          Expanded(
                            child: MaterialButton(
                              onPressed: () async {
                                onIconPressed();
                                await widget.searchByMyPos();
                              },
                              color: Colors.white,
                              child: Text(
                                "Рядом",
                                style: TextStyle(
                                  color: Color(0xFF29304a),
                                  fontSize: 16,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ),
                          ),
                          SizedBox(
                            width: 20,
                          ),
                          Expanded(
                            child: OutlineButton(
                              onPressed: () async {
                                onIconPressed();
                                await widget.searchByAddress();
                              },
                              borderSide: BorderSide(
                                color: Color(0xFF23eacb),
                              ),
                              color: Theme.of(context).primaryColor,
                              child: Text(
                                "По адресу",
                                style: TextStyle(
                                  color:
                                      Theme.of(context).scaffoldBackgroundColor,
                                  fontSize: 16,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
              Align(
                alignment: Alignment(0, 0.9),
                child: GestureDetector(
                  onTap: () {
                    onIconPressed();
                  },
                  child: ClipPath(
                    clipper: CustomMenuClipper(),
                    child: Container(
                      width: 100,
                      height: 50,
                      color: Color(0xFF29304a),
                      alignment: Alignment.center,
                      child: AnimatedIcon(
                        progress: _animationController.view,
                        icon: AnimatedIcons.menu_close,
                        color: Color(0xFF1BB5FD),
                        size: 25,
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

class CustomMenuClipper extends CustomClipper<Path> {
  @override
  Path getClip(Size size) {
    Paint paint = Paint();
    paint.color = Colors.white;

    Path path = Path();
    path.moveTo(0, 0);

    path.addRRect(RRect.fromRectAndCorners(
        Rect.fromPoints(Offset(0, 0), Offset(100, 50)),
        bottomLeft: Radius.circular(10),
        bottomRight: Radius.circular(10)));

    path.close();
    return path;
  }

  @override
  bool shouldReclip(CustomClipper<Path> oldClipper) {
    return true;
  }
}
