import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart';
import 'package:parking_finder/model/place.dart';
import 'package:parking_finder/model/lat_lon.dart';
import 'package:parking_finder/ui/components/divider_with_text_widget.dart';

import 'package:auto_size_text/auto_size_text.dart';
import 'package:parking_finder/util/constants.dart';
import 'dart:async';
import 'package:uuid/uuid.dart';

const kDaDataApiKey = "a76e35d839b20faf08921dd7a0f3d4796ac3ce33";
const kGeocodingUrl = "https://cleaner.dadata.ru/api/v1/clean/address";
const kXSecret = "980f55407af13644bc81b850e5bb81818dc5a117";
const kContentType = "application/json";
const kAuthorization = "Token a76e35d839b20faf08921dd7a0f3d4796ac3ce33";
class SearchLocationView extends StatefulWidget {
  SearchLocationView({Key key,this.updateLocation}) : super(key: key);
  final Function updateLocation;
  @override
  _SearchLocationViewState createState() => _SearchLocationViewState();
}

class _SearchLocationViewState extends State<SearchLocationView> {
  TextEditingController _searchController = new TextEditingController();

  var uuid = new Uuid();

  String _sessionToken;

  String _heading;
  List<Place> _placesList;
  final List<Place> _suggestedList = [];
  LatLon latLon;

  @override
  void initState() {
    super.initState();
    _heading = "Suggestions";
    _placesList = _suggestedList;
    _searchController.addListener(_onSearchChanged);

  }

  @override
  void dispose() {
    _searchController.removeListener(_onSearchChanged);
    _searchController.dispose();
    super.dispose();
  }

  _onSearchChanged() {
    if(_sessionToken == null) {
      setState(() {
        _sessionToken = uuid.v4();
        print("uuid = $_sessionToken");
      });
    }
    getLocationResults(_searchController.text);
  }

  void getLocationResults(String input) async {
    if (input.isEmpty) {
      setState(() {
        _heading = "Пусто";
      });
      return;
    }

    String baseURL = 'https://suggestions.dadata.ru/suggestions/api/4_1/rs/suggest/address';
    String request = '$baseURL?query=$input&token=$kDaDataApiKey';
    Response response = await get(request);

    final predictions = jsonDecode(response.body)['suggestions'];

    List<Place> _displayResults = [];

    for (var i=0; i < predictions.length; i++) {
      String name = predictions[i]['value'];
      _displayResults.add(Place(name));
    }

    if (this.mounted) {
      setState(() {
        _heading = "Результа";
        _placesList = _displayResults;
      });
    }
  }



  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Color(0xFF29304a),
      ),
      backgroundColor: Color(0xFF29304a),
      body: Center(
        child: Column(
          children: <Widget>[
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Container(
                alignment: Alignment.centerLeft,
                decoration: kBoxDecorationStyle,
                height: 60.0,
                child: TextField(
                  controller: _searchController,
                  style: TextStyle(
                    color:  Color(0xFF29304a),
                    fontFamily: 'OpenSans',
                  ),
                  decoration: InputDecoration(
                    border: InputBorder.none,
                    contentPadding: EdgeInsets.only(top: 14.0),
                    prefixIcon: Icon(
                      Icons.search,
                      color: Color(0xFF29304a),
                    ),
                    hintText: 'Введите адрес',
                    hintStyle: kHintTextStyle,
                  ),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(left: 10.0, right: 10.0,bottom: 14),
              child: DividerWithText(
                dividerText: _heading
              ),
            ),
            Expanded(
              child: ListView.builder(
                itemCount: _placesList.length,
                itemBuilder: (BuildContext context, int index) =>
                    buildPlaceCard(context, index),
              ),
            ),
          ],
        ),
      ),
    );
  }
  Widget buildPlaceCard(BuildContext context, int index) {
    return Hero(
      tag: "SelectedTrip-${_placesList[index].name}",
      transitionOnUserGestures: true,
      child: Container(
        child: Padding(
          padding: const EdgeInsets.only(
            left: 8.0,
            right: 8.0,
          ),
          child: Card(
            child: InkWell(
              child: Row(
                children: <Widget>[
                  Expanded(
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        children: <Widget>[
                          Row(
                            children: <Widget>[
                              Flexible(
                                child: AutoSizeText(_placesList[index].name,
                                    maxLines: 3,
                                    style: TextStyle(fontSize: 16.0)),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),

                ],
              ),
              onTap: () async {
                List<String>jsonMap = List();
                jsonMap.add("${_placesList[index].name}");
                var responseBody = await apiRequest(kGeocodingUrl, jsonMap);
                setState(() {
                  _sessionToken = null;
                });
                await widget.updateLocation(LatLon(lat: double.parse(responseBody[0]['geo_lat']), lon: double.parse(responseBody[0]['geo_lon'])));
                Navigator.pop(context);
              },
            ),
          ),
        ),
      ),
    );
  }
}

Future apiRequest(String url, List<String> jsonMap) async {
  Client client = Client();
  Response response = await client.post(
      kGeocodingUrl,
      body: jsonEncode(jsonMap),
      headers: {
        'Content-type': kContentType,
        'Authorization': kAuthorization,
        'X-Secret': kXSecret
      });
  if(response.statusCode == 200){
    var decodeData = jsonDecode(response.body);
    return decodeData;
  }else{
    print(response.statusCode);
  }
  client.close();
}
