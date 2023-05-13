import utils

def main():
    day_type_user='1d'
    mark_type_user='Close'
    model_name_user='MC'
    tickerSymbol='META'
        
    df, tickerSymbol = utils.load_data_yahoo(tickerSymbol, period=day_type_user)
    df=utils.calc_log_return(df,mark_type=mark_type_user)
    n, fcast_dates=utils.pred_interval(data=df,day_type=day_type_user)
    
    utils.plot_logr_price(data=df,tickerSymbol=tickerSymbol,mark_type=mark_type_user)
    print(utils.descriptive_stats(df))

    #Calling models
    if model_name_user == 'MC':
        mc_mid_pred, min_values_mc, max_values_mc=utils.mc_model(data=df,sim=10000, n=n, day_type=day_type_user, mark_type=mark_type_user)
        utils.plot_predicted(data=df, mid=mc_mid_pred, lower=min_values_mc, upper=max_values_mc,fcast_dates=fcast_dates,tickerSymbol=tickerSymbol,mark_type=mark_type_user,model_name=model_name_user)
    elif model_name_user == 'ARMA':
        pass
    elif model_name_user == 'GARCH':
        pass
    elif model_name_user == 'ARMA-GARCH':
        pass
    elif model_name_user == 'LSTM':
        pass
    else:
        print('Not valid model')

if __name__ == "__main__":
    main()
