import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'package:flutter/cupertino.dart';
import 'package:http/http.dart';
import 'package:parking_finder/model/parking.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:typed_data';

class NetworkHelper{


  final contentType = "application/json";


  NetworkHelper({@required this.url,this.tokenAuth});
  final String url;
  final tokenAuth;

  Future<bool> authorization() async{
    SharedPreferences prefs = await SharedPreferences.getInstance();
    String token = prefs.getString('token');
    Response response = await get(url,headers: {
      'Content-type': contentType,
      'x-access-token value': token
    });
    if(response.statusCode == 200){
      print("status = ${response.statusCode}");
      return true;
    }else{
      print("status = ${response.statusCode}");
      return false;
    }
  }

  Future logining(String username,String password) async {
    Client client = Client();

    String basicAuth =
        'Basic ' + base64Encode(utf8.encode('$username:$password'));
    print(basicAuth);

    Response response = await client.get(url,
        headers: <String, String>{'authorization': basicAuth});
    print(response.statusCode);
    print(response.body);
    if(response.statusCode == 200){
      var decodeData = jsonDecode(response.body);
      //TODO раскомитеть преференс
      //SharedPreferences prefs = await SharedPreferences.getInstance();
      //prefs.setString('token', decodeData['token']);
      return decodeData;
    }else{
      print(response.statusCode);
    }
    client.close();
  }

  Future<List<ParkingPlace>> loadCoord(Map<String, String> jsonMap) async{
    Client client = Client();
    List<ParkingPlace> listParkingPlace = List<ParkingPlace>();
    Response response = await client.post(
        url,
        body: json.encode(jsonMap),
        headers: {'Content-type': contentType}
    );
    print(response.statusCode);
    print(response.body);
    if(response.statusCode == 200){
      var decodeData = jsonDecode(response.body);

      print("decodeData = " + decodeData['lat']);
      var rng = new Random();
      listParkingPlace.add(ParkingPlace(id: rng.nextInt(100),lat:double.parse( decodeData['lat']),lon: double.parse(decodeData['lon']), baseImage: decodeData['ImageBytes']));
      return listParkingPlace;
    }else{
      print("jsonMap = " + jsonMap['lon']);
      print("false  =  ${response.statusCode}");
    }
    client.close();
  }

}