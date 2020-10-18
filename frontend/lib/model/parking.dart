import 'package:flutter/cupertino.dart';
import 'package:parking_finder/service/location.dart';

class ParkingPlace{
  ParkingPlace({@required this.id,@required this.lat, @required this.lon, this.baseImage});
  int id;
  double lat;
  double lon;
  String baseImage;
}