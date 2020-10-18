import 'package:flutter/material.dart';
import 'package:parking_finder/ui/screens/login_screen.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Parking Finder',
      debugShowCheckedModeBanner: false,
      home: LoginScreen(),
    );
  }
}