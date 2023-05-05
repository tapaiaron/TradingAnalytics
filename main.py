import modules_functions

def main():
    df=modules_functions.calc_log_return(modules_functions.load_data_yahoo(tickerSymbol='META'), period='1d')
    modules_functions.plot_logr_price(df)
    print(modules_functions.descriptive_stats(df))
    modules_functions.mc_model(df, n, day_type="1d")
    modules_functions.plot_predicted(model=modules_functions.mc_model(data, n),interval=pred_interval())

if __name__ == "__main__":
    main()
