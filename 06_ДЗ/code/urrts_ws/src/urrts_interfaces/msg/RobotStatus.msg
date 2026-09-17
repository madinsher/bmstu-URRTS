# Status of one robot for monitoring and metrics (NOT used for coordination).
string robot
string state             # executor FSM state
string order_id          # current order, empty if none
float64 x
float64 y
float64 battery          # 0..1, reserved for extensions
builtin_interfaces/Time stamp
