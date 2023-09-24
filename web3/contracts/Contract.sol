// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

contract DeAcc{
    struct Accident {
        string loc;
        string date;
        string time;
        string snapShot;
        string plate;
    }

    struct userAccident {
        string snapShot;
        string lat;
        string long;
        address user;
    }

    struct Insurance {
        string name;
        string phone;
        uint256 blockNo;
        address user;
    }

    mapping(uint256 => Accident) public accidents;
    mapping(uint256 => userAccident) public userAccidents;

    mapping(uint256 => Insurance) public insurances;

    uint256 public numberOfAccidents = 0;
    uint256 public numberOfUserAccidents = 0;

    function addAccident(string memory _loc, string memory _date, string memory _time, string memory _snapShot, string memory _plate) public {
        numberOfAccidents++;
        accidents[numberOfAccidents] = Accident(_loc, _date, _time, _snapShot, _plate);
    }

    function userAddsAccident(string memory _snapShot, string memory _lat, string memory _long) public {
        numberOfUserAccidents++;
        userAccidents[numberOfUserAccidents] = userAccident(_snapShot, _lat, _long, msg.sender);
    }

    function getAccidents() public view returns(Accident[] memory) {
        Accident[] memory _accidents = new Accident[](numberOfAccidents);

        for (uint256 i = 0; i < numberOfAccidents; i++) {
            _accidents[i] = accidents[i];
        }

        return _accidents;
    }

    function getUserAccidents() public view returns(userAccident[] memory) {
        userAccident[] memory _userAccidents = new userAccident[](numberOfUserAccidents);

        for (uint256 i = 0; i < numberOfUserAccidents; i++) {
            _userAccidents[i] = userAccidents[i];
        }

        return _userAccidents;
    }

    function getAccident(uint256 _id) public view returns(string memory, string memory, string memory, string memory, string memory) {
        return (accidents[_id].loc, accidents[_id].date, accidents[_id].time, accidents[_id].snapShot, accidents[_id].plate);
    }

    function getUserAccident(uint256 _id) public view returns(string memory, string memory, string memory, address) {
        return (userAccidents[_id].snapShot, userAccidents[_id].lat, userAccidents[_id].long, userAccidents[_id].user);
    }

    function reqInsurance(string memory _name, string memory _phone, uint256 _blockNo) public {
        insurances[_blockNo] = Insurance(_name, _phone, _blockNo, msg.sender);
    }

    function getInsurance(uint256 _blockNo) public view returns(string memory, string memory, uint256, address) {
        return (insurances[_blockNo].name, insurances[_blockNo].phone, insurances[_blockNo].blockNo, insurances[_blockNo].user);
    }

    function getNumberOfAccidents() public view returns(uint256) {
        return numberOfAccidents;
    }
}