#ifndef RUSTCORE_BRIDGE_H
#define RUSTCORE_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

char* train_model(const char* filename);
void free_string(char* s);

#ifdef __cplusplus
}
#endif

#endif // RUSTCORE_BRIDGE_H
