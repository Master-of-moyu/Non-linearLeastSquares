#ifndef _LOG_H
#define _LOG_H

#include <ctime>
#include <iostream>
#include <cstring>
#ifdef ANDROID
#include <android/log.h>
#endif

#ifndef log_debug
#define log_debug(format, ...) \
    printf("\033[0;35m[LOG-DEBUG]\033[0;0m " format "\n", ##__VA_ARGS__)
#endif

#ifndef log_error
#define log_error(format, ...) \
    printf("\033[0;31m[LOG-ERROR]\033[0;0m " format "\n", ##__VA_ARGS__)
#endif

#ifndef log_info
#define log_info(format, ...) \
    printf("\033[0;32m[LOG-INFO]\033[0;0m " format "\n", ##__VA_ARGS__)
#endif

#ifndef log_warn
#define log_warn(format, ...) \
    printf("\033[0;33m[LOG-WARN]\033[0;0m " format "\n", ##__VA_ARGS__)
#endif

#ifndef PRINT_VAR
#ifndef NDEBUG
#define PRINT_VAR(var) std::cout << #var << " = " << var << std::endl
#else
#define PRINT_VAR(var)
#endif
#endif

#endif
