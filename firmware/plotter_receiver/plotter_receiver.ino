/**
 * plotter_receiver — ESP32 WiFi endpoint for the 5-Bar SCARA plotter
 *
 * Connects to your LAN, starts an HTTP server on SERVER_PORT, and waits for
 * the Raspberry Pi to POST stroke JSON to /plot.  Fill in WIFI_SSID and
 * WIFI_PASSWORD below, flash to the ESP32, then copy the printed IP address
 * into the Pi's PLOTTER_IP environment variable.
 *
 * Dependencies (install via Arduino IDE → Library Manager):
 *   • ArduinoJson  ≥ 6.21  (v7 also works — see note in handle_plot)
 *
 * Built-in ESP32 Arduino core libraries used (core ≥ 2.x):
 *   WiFi, WebServer
 *
 * Stroke JSON format sent by the Pi (process.py):
 *   { "width": <px>, "height": <px>,
 *     "paths": [ [[x,y], [x,y], ...], ... ] }
 *   Coordinates are image-pixel space, origin top-left, y-down.
 *   Step 3 (IK + motor driver) maps these into workspace coords.
 */

#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>

// ── User configuration ────────────────────────────────────────────────────────
#define WIFI_SSID     "YOUR_SSID"
#define WIFI_PASSWORD "YOUR_PASSWORD"
#define SERVER_PORT   9000          // must match PLOTTER_PORT on the Pi (default 9000)

// JSON document heap budget.  Each point costs ~40 bytes in the document tree.
// Reduce max_dim or raise min_length in process.py Params if you hit resets.
#define JSON_BUDGET   (96 * 1024)   // 96 KB — handles ~2 000 strokes at default settings
// ─────────────────────────────────────────────────────────────────────────────

WebServer server(SERVER_PORT);

// ── Stroke callback ───────────────────────────────────────────────────────────
// Called once per /plot POST with the fully-parsed document.
// Replace the body with IK + motor-driver logic in Step 3.
void on_plot(JsonDocument& doc) {
    int img_w  = doc["width"]  | 0;
    int img_h  = doc["height"] | 0;
    JsonArray paths = doc["paths"];

    Serial.printf("[plot] %d x %d px image, %d strokes, free heap %lu B\n",
                  img_w, img_h, (int)paths.size(), (unsigned long)ESP.getFreeHeap());

    for (JsonArray stroke : paths) {
        // stroke is an ordered list of [x, y] pixel-space points.
        // TODO: transform into workspace coords, run IK, stream motor moves.
        for (JsonArray pt : stroke) {
            float x = pt[0];
            float y = pt[1];
            (void)x; (void)y;   // suppress unused warnings until Step 3 is wired up
        }
    }
}
// ─────────────────────────────────────────────────────────────────────────────

void handle_plot() {
    if (!server.hasArg("plain")) {
        server.send(400, "application/json", "{\"ok\":false,\"error\":\"no body\"}");
        return;
    }

    // ArduinoJson v6: DynamicJsonDocument doc(JSON_BUDGET);
    // ArduinoJson v7: replace with  JsonDocument doc;  (no size arg)
    DynamicJsonDocument doc(JSON_BUDGET);
    DeserializationError err = deserializeJson(doc, server.arg("plain"));
    if (err) {
        String msg = String("{\"ok\":false,\"error\":\"") + err.c_str() + "\"}";
        server.send(400, "application/json", msg);
        Serial.printf("[plot] parse error: %s\n", err.c_str());
        return;
    }

    // Respond before processing — keeps the Pi from timing out on large jobs.
    server.send(200, "application/json", "{\"ok\":true}");
    on_plot(doc);
}

void handle_status() {
    char buf[128];
    snprintf(buf, sizeof(buf),
             "{\"ok\":true,\"ip\":\"%s\",\"heap\":%lu}",
             WiFi.localIP().toString().c_str(),
             (unsigned long)ESP.getFreeHeap());
    server.send(200, "application/json", buf);
}

// ── WiFi helpers ──────────────────────────────────────────────────────────────
static void wifi_connect() {
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.print("[wifi] connecting to " WIFI_SSID " ");
    unsigned long t = millis();
    while (WiFi.status() != WL_CONNECTED) {
        if (millis() - t > 20000) {
            Serial.println("\n[wifi] timed out — will retry in loop");
            return;
        }
        delay(500);
        Serial.print(".");
    }
    Serial.println();
    Serial.print("[wifi] connected  IP: ");
    Serial.println(WiFi.localIP());
    Serial.printf("[wifi] >>> set PLOTTER_IP=%s on the Pi to enable forwarding <<<\n",
                  WiFi.localIP().toString().c_str());
}

void setup() {
    Serial.begin(115200);
    Serial.println("\n[plotter-receiver] booting …");

    wifi_connect();

    server.on("/plot", HTTP_POST, handle_plot);
    server.on("/",     HTTP_GET,  handle_status);
    server.begin();
    Serial.printf("[http] server listening on port %d\n", SERVER_PORT);
}

void loop() {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[wifi] lost connection, reconnecting …");
        WiFi.disconnect();
        wifi_connect();
    }
    server.handleClient();
}
