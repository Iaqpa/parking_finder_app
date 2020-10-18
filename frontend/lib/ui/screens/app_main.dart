import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:parking_finder/model/lat_lon.dart';
import 'package:parking_finder/model/parking.dart';
import 'package:parking_finder/service/networking.dart';
import 'package:parking_finder/ui/components/bottom_sheet_shape.dart';
import 'package:parking_finder/service/location.dart';
import 'package:parking_finder/ui/sidebar/sidebar.dart';
import 'location_view.dart';


enum SearchLocationType{myPosition,address,city}

class AppScreen extends StatefulWidget {
  @override
  _AppScreenState createState() => _AppScreenState();
}


class _AppScreenState extends State<AppScreen> {
  Map<String,String> jsonMap;
  List<ParkingPlace> listParkingPlace = List<ParkingPlace>();

  final Map<String, Marker> _markers = {};


  Completer<GoogleMapController> _controller = Completer();
  LatLon latLon;

  double zoom;
  CameraPosition _kCameraPosition = CameraPosition(
    target: LatLng(55.787519, 49.123687),
    zoom: 12,
  );



  void updateLatLon (LatLon locationData)async{
    setState(() {
      if(locationData== null){
        return;
      }
      latLon =locationData;
      jsonMap = {'lon': '${latLon.lon}', 'lat': '${latLon.lat}'};
    });
    NetworkHelper networkHelper = NetworkHelper(url: "https://ee0d0fba02f0.ngrok.io/api/v1.0/coords");
    List<ParkingPlace> transit= await networkHelper.loadCoord(jsonMap);
    setState(() {
      listParkingPlace = transit;
    });
  }
  @override
  void initState() {
    super.initState();
  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: <Widget>[
         GoogleMap(
           mapType: MapType.hybrid,
           initialCameraPosition: _kCameraPosition,
           onMapCreated: (GoogleMapController controller) {
             _controller.complete(controller);
           },
           markers: _markers.values.toSet(),
            ),
          SideBar(searchByMyPos: searchByMyPos,searchByAddress: searchByAddress,),
        ],
      ),
      //floatingActionButton: FloatingActionButton(
      //  onPressed:(){
      //    _openSignOutDrawer();
      //  },
      //backgroundColor: Color(0xFF29304a),
      //child: Icon(Icons.search,
      //      color: Colors.white),
      //),

      );
  }

  Future<void> searchPlace() async {
    final GoogleMapController controller = await _controller.future;
    controller.animateCamera(CameraUpdate.newCameraPosition(CameraPosition(
        target: LatLng(latLon.lat, latLon.lon),
        tilt: 59.440717697143555,
        zoom: 16)
    ));
  }

  Future<void> _onMapCreated() async {
    setState(() {
      _markers.clear();
      for (ParkingPlace parkingPlaceItem in listParkingPlace) {
        final marker = Marker(
          onTap:(){
            _openSignOutDrawer();
          },
          markerId: MarkerId(parkingPlaceItem.id.toString()),
          position: LatLng(parkingPlaceItem.lat, parkingPlaceItem.lon),
          infoWindow: InfoWindow(
            title: "Парковка",

          ),
        );
        _markers[parkingPlaceItem.id.toString()] = marker;
      }
    });
  }

  Future<void> searchByMyPos()async{
    Location location = Location();
    await location.getCurrentLocation();
    NetworkHelper networkHelper = NetworkHelper(url: "https://ee0d0fba02f0.ngrok.io/api/v1.0/coords");
    setState(() {
      latLon = LatLon(lat: location.latitude, lon: location.longitude);
      jsonMap = {'lon': '${location.longitude}', 'lat': '${location.latitude}'};
    });
    List<ParkingPlace> transit= await networkHelper.loadCoord(jsonMap);
    setState(() {
      listParkingPlace = transit;
    });
    await searchPlace();
    await _onMapCreated();
  }

  Future<void> searchByAddress()async{
    await Navigator.push(context, MaterialPageRoute(builder: (context){
      return SearchLocationView(updateLocation: updateLatLon,);
    }));
    await searchPlace();
    await _onMapCreated();
  }
  void _openSignOutDrawer() {
    showModalBottomSheet(
        shape: BottomSheetShape(),
        backgroundColor: Color(0xFF29304a),
        context: context,
        builder: (context) {
          return Container(
            padding: const EdgeInsets.only(
              top: 16,
              bottom: 16,
              left: 24,
              right: 24,
            ),
            height: 300,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: <Widget>[
                Container(
                  height: 200,
                  child: Image.network("https://ee0d0fba02f0.ngrok.io/api/v1.0/get_image"),
                ),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  mainAxisSize: MainAxisSize.max,
                  children: <Widget>[
                    Expanded(
                      child: MaterialButton(
                        onPressed: () async {

                        },
                        color: Colors.white,
                        child: Text(
                          "Занять",
                          style: TextStyle(
                            color: Color(0xFF29304a),
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                    ),
                    SizedBox(
                      width: 10,
                    ),
                    Expanded(
                      child: OutlineButton(
                        onPressed: () {
                        },
                        borderSide: BorderSide(
                          color: Color(0xFF23eacb),
                        ),
                        color: Theme.of(context).primaryColor,
                        child: Text(
                          "Занято",
                          style: TextStyle(
                            color: Theme.of(context).scaffoldBackgroundColor,
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                    ),
                    SizedBox(
                      width: 10,
                    ),
                    Expanded(
                      child: OutlineButton(
                        onPressed: () {
                        },
                        borderSide: BorderSide(
                          color: Color(0xFF23eacb),
                        ),
                        color: Theme.of(context).primaryColor,
                        child: Text(
                          "Ошибка",
                          style: TextStyle(
                            color: Theme.of(context).scaffoldBackgroundColor,
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
          );
        });
  }

}



