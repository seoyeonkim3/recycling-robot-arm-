#define trig 9       // Ultrasonic sensor trigger pin
#define echo 8       // Ultrasonic sensor echo pin
#define signal 4     // Proximity sensor pin

void setup() {
  Serial.begin(9600); // Initialize serial communication
  pinMode(trig, OUTPUT); // Set trigger pin as output
  pinMode(echo, INPUT);  // Set echo pin as input
  pinMode(signal, INPUT); // Set proximity sensor pin as input
}

void loop() {
  // Read data from ultrasonic sensor
  long duration, distance;
  digitalWrite(trig, LOW); // Ensure trigger pin is low
  delayMicroseconds(2);    // Wait for 2 microseconds
  digitalWrite(trig, HIGH); // Send a 10us pulse
  delayMicroseconds(10);
  digitalWrite(trig, LOW); // Reset trigger pin
  duration = pulseIn(echo, HIGH); // Measure the pulse duration
  distance = duration * 170 / 1000;  // Calculate distance in mm
  Serial.print("D:"); // Add a tag for ultrasonic data
  Serial.println(distance);

  // Read data from proximitSSy sensor
  int proximity = !digitalRead(signal);  // Invert proximity sensor value (0 or 1)
  Serial.print("P:"); // Add a tag for proximity data
  Serial.println(proximity);

  delay(100); // Delay between data transmissions
}
